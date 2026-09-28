#!/usr/bin/env bash
# Opt-in hidden-state capture restart for the b12x stack. NOT run by this commit.
#
# Designed to be launched detached so its verdict survives the terminal that launched it (and an
# interrupted Copilot chat):
#
#   setsid ./restart-hidden-capture.sh &          # detached, survives terminal close
#   setsid nohup ./restart-hidden-capture.sh >~/hidden_restart.out 2>&1 < /dev/null &
#
# Writes a machine-parsable verdict to $OUT (~ by default, or given as arg1) that stays readable
# even if this shell/session was lost, plus a human timestamped log ($LOG). It only touches
# serve.sh + this one container; no gateway (:8088), no model, no recipe, no MTP.
#
# OFF by default: serve.sh WITHOUT HIDDEN_CAPTURE=1 is byte-for-byte the serving path that runs today,
# so a restart done with this flag unset changes nothing.
#
# Exit 0 + "restart=ok" means: serve.sh returned 0 (serve.sh itself validates the read route), then
# /health answered, then a real request's captured vector is readable via the /flashnext route.
# Any failure exits non-zero and the last line in $OUT is "restart=failed reason=<short>".
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SERVE="$HERE/serve.sh"
CONTAINER="${CONTAINER:-qwen38-flash}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
WARMUP_SECONDS="${WARMUP_SECONDS:-10}"   # fixed wait before polling :$API_PORT
READY_TIMEOUT="${READY_TIMEOUT:-600}"    # ceiling AFTER warm-up (load is minutes)
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="${LOG:-$HOME/hidden_restart_$STAMP.log}"
OUT="${1:-$HOME/hidden_restart_$STAMP.out}"

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG"; }
verdict() {                       # last line wins, append so the outcome file is readable offline
  printf '%s\n' "$*" >>"$OUT"
  if [ "${1%%=*}" = restart ] && [ "${1#restart=}" != ok ]; then :; fi
}
RESTART_STARTED=0
restore_baseline() {
  [ "$RESTART_STARTED" = 1 ] || return 0
  log "restoring the previous serving mode with HIDDEN_CAPTURE=0"
  if BACKEND=b12x HIDDEN_CAPTURE=0 bash "$SERVE" -d >>"$LOG" 2>&1; then
    local deadline=$(( $(date +%s) + READY_TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
      if curl -fsS --max-time 5 "http://$API_HOST:$API_PORT/health" >/dev/null 2>&1; then
        verdict "rollback=ok mode=baseline"
        log "baseline serving restored"
        return 0
      fi
      sleep 3
    done
  fi
  verdict "rollback=failed"
  log "baseline serving could not be confirmed; inspect container logs"
  return 1
}
fail() {
  log "FAIL: $*"
  verdict "restart=failed reason=$1"
  restore_baseline || true
  exit 1
}

# Outcome file starts clean (one verdict per run) and the log keeps its history.
: >"$OUT"
echo "restart-hidden-capture $(date '+%Y-%m-%d %H:%M:%S') pid=$$ log=$LOG out=$OUT" >>"$LOG"
echo "restart=running" >>"$OUT"

if [ ! -f "$SERVE" ]; then fail "servescript-missing"; fi
if [ ! -f "$HERE/flashnext-int4-b12x/hidden_state.py" ]; then fail "mod-missing"; fi

# Copy only this fork's mod and recipes BEFORE touching the running container.
# This does not inspect, pull, tag, or build any Docker image, or update Eugr.
log "syncing mod and recipes from this checkout before stopping the model"
if ! "$HERE/eugr-setup.sh" --sync-only >>"$LOG" 2>&1; then fail "mod-install"; fi

BEFORE="$(docker ps -a --format '{{.Names}} {{.Status}}' --filter "name=^${CONTAINER}$" 2>/dev/null || true)"
log "container before: '${BEFORE:-none}'"
if printf '%s' "$BEFORE" | grep -qE "^${CONTAINER} Up"; then
  # Check the running image before stopping it. A mismatch leaves Qwen alive.
  log "checking vLLM patch anchors and InputBatch fields (read-only)"
  if ! docker exec "$CONTAINER" python3 -c '
from pathlib import Path
root = Path("/usr/local/lib/python3.12/dist-packages/vllm")
runner = (root / "v1/worker/gpu/model_runner.py").read_text()
routers = (root / "entrypoints/launchers/api_server/routers.py").read_text()
anchor = "    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:\n        shard_metadata = None"
checks = {
    "sample-anchor": runner.count(anchor) == 1,
    "route-anchor": routers.count("    register_vllm_serve_api_routers(app)\n") == 1,
}
for name in ("is_prefilling_np", "num_computed_tokens_np", "seq_lens_cpu_upper_bound",
             "prefill_len_np", "cu_num_logits_np", "logits_indices"):
    checks[name] = name in runner
for name, ok in checks.items():
    print(f"{name}: {ok}")
if not all(checks.values()):
    raise SystemExit(1)
' >>"$LOG" 2>&1; then fail "runtime-anchor-mismatch"; fi
  log "runtime anchors present; stopping running container"
  RESTART_STARTED=1
  docker stop "$CONTAINER" >>"$LOG" 2>&1 || true
fi

log "BACKEND=b12x serve.sh -d HIDDEN_CAPTURE=1 (pinned Eugr recipe, detached, route probe)"
# serve.sh with HIDDEN_CAPTURE=1 refuses to report success unless the read route answers 422/400/200 etc.
# On a missing router/anchor it prints FATAL and exits 1 -- captured here and re-reported.
# NOTE: NOT exec, so we learn serve's verdict.
cd "$HERE"
BACKEND=b12x HIDDEN_CAPTURE=1 bash ./serve.sh -d >>"$LOG" 2>&1
SERVE_RC=$?
log "serve.sh exit=$SERVE_RC"
if [ "$SERVE_RC" != 0 ]; then
  log "serve.sh failed. Inspect its patch and route output above."
  fail "serve-rc-$SERVE_RC"
fi
log "serve.sh OK: it saw the read route registered"

log "fixed warm-up ${WARMUP_SECONDS}s before polling :$API_PORT"
sleep "$WARMUP_SECONDS"

log "polling /health up to ${READY_TIMEOUT}s after warm-up"
end=$(( $(date +%s) + WARMUP_SECONDS + READY_TIMEOUT ))
health=0
while [ "$(date +%s)" -lt "$end" ]; do
  if curl -fsS --max-time 5 "http://$API_HOST:$API_PORT/health" >>"$LOG" 2>&1; then health=1; break; fi
  sleep 3
done
[ "$health" = 1 ] || fail "health-timeout"
log "health = 200 OK"

# serve.sh already proved the route exists; this is an independent second look: a bare request without
# req_id must NOT be 404 (200/422/400 all mean FastAPI owns the path and it registered).
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "http://$API_HOST:$API_PORT/flashnext/hidden_state/read?" 2>/dev/null)" || code=000
log "bare GET /flashnext/hidden_state/read -> HTTP $code"
[ "$code" != 404 ] && [ "$code" != 000 ] || fail "route-404"
log "read route confirmed (not a 404)"

log "smoke: one chat completion, then read back its captured vector"
RESP="$(curl -fsS --max-time 180 "http://$API_HOST:$API_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED_NAME:-qwen3.8-flash-next-a5b}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1,\"temperature\":0}" 2>>"$LOG" || true)"
[ -n "$RESP" ] || fail "chat-no-response"
REQ_ID="$(printf '%s' "$RESP" | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')"
[ -n "$REQ_ID" ] || fail "chat-id-missing"
log "chat id: $REQ_ID"

capture=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  body="$(curl -fsS --max-time 5 "http://$API_HOST:$API_PORT/flashnext/hidden_state/read?req_id=$REQ_ID" 2>>"$LOG" || true)"
  if [ -n "$body" ]; then capture="$body"; break; fi
  sleep 1
done
# If the vector never arrives: capture disabled (unexpected layout) or prefill rule gap -- not ambiguous.
if [ -z "$capture" ]; then
  tail2="$(docker logs --tail 5 "$CONTAINER" 2>&1 | grep -a 'hidden-state capture disabled' | tail -1 || true)"
  [ -z "$tail2" ] || log "worker log says: $tail2"
  fail "capture-empty"
fi
hs="$(printf '%s' "$capture" | grep -oE '"hidden_size"[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+$')"
log "captured req_id=$REQ_ID hidden_size=${hs:-unknown}"
[ "$hs" = 2560 ] || fail "hidden-size-mismatch"

verdict "restart=ok container=$CONTAINER route=live req=$REQ_ID hidden_size=${hs:-unknown} log=$LOG"
log "DONE. Everything outside this container (gateway, recipe, MTP) is untouched. To turn OFF: unset HIDDEN_CAPTURE and run ./serve.sh -d"
exit 0
