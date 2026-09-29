# SystemOne come nucleo decisionale di gioco, computer-use e agente autonomo

Stato: progetto architetturale per `feature/visual-systemone`, 2026-09-29. Non introduce un secondo modello né modifica il serving Eugr/B12X. Le componenti descritte come *da costruire* non sono tool già esposti dal repository.

## 1. Confine tra modello, decisione e azione

```text
Sorgente evento / osservazione
    -> Orchestratore: obiettivo, memoria breve, stato, azioni ammesse, scadenza
        -> SystemOne :8088 per una scelta tipizzata
        -> Qwen chat :8000 per pianificazione, spiegazione o uso flessibile di tool
    -> Esecutore locale (gioco, desktop, browser, app)
    -> Nuova osservazione e verifica esito
    -> Log, memoria persistente e trigger successivo
```

Il DGX Spark può ospitare l'unica istanza Qwen e il gateway. Gli strumenti che osservano e agiscono devono vivere dove gira l'ambiente: per esempio sul PC Windows con la RTX 5080 per Paint o un gioco locale. L'orchestratore può risiedere sul PC o su Spark, usando gli endpoint già raggiungibili attraverso la rete privata.

Con `state_id` omesso o `null`, `/v1/systemone` non legge un frame e invia a Qwen lo **stato testuale** costruito dai tool, dall'utente o da un precedente passo di Qwen. Non sopprime la richiesta: salta solo la parte visiva. Con `state_id` presente, il frame JPEG corrente della sessione è obbligatorio. Il gateway non osserva da sé lo schermo né richiama strumenti autonomamente.

### Inventario verificato nel fork

| Disponibile ora | Funzione |
| --- | --- |
| Qwen vLLM `:8000/v1/chat/completions` | Generazione, visione e tool calling se un harness fornisce e gestisce realmente i tool. |
| Gateway `:8088/v1/systemone` | `state` + domande `choice`, `noul`, `score`; immagine tramite `state_id` oppure solo testo senza `state_id`. Una chiamata vLLM per domanda. |
| `POST /v1/vision/{state_id}/frame` | Sostituisce il JPEG della sessione; nessuna storia di frame. |
| `/flashnext/hidden_state/read` e `/trace` | Diagnostica opt-in del runner corrente; non servono all'esecuzione delle azioni. |
| Script `push_frame.py`, probe, test e confronti | Test manuali, non un driver per giochi o desktop. |

Limiti del gateway attuale: stato testuale massimo 2048 caratteri; fino a 4 domande, 2–8 opzioni per `choice`/`score`; frame JPEG massimo 300 KB, validità 10 secondi, 8 sessioni. La `confidence` misura la concentrazione delle probabilità tra opzioni, **non** la probabilità che l'azione riesca. Sulla build vLLM corrente abbiamo visto almeno un HTTP 500 durante la formattazione dei logprob: va gestito e verificato prima di un ciclo senza supervisione. La modalità solo testo è nel branch; l'attivazione sul gateway Spark dipende dal suo riavvio, non da quello del modello.

## 2. Scelta tra decisione rapida e generazione

| Usa SystemOne | Usa Qwen chat |
| --- | --- |
| L'osservazione è già sufficiente e le azioni valide sono finite e definite. | Serve definire un obiettivo, fare un piano, cercare tool o interpretare un errore nuovo. |
| Bisogna scegliere, stimare urgenza o classificare un evento. | Serve scrivere testo, codice, una spiegazione o una sequenza nuova. |
| Un'azione deve rispettare una scadenza breve; esiste una scelta prudente come `wait` o `release`. | Non c'è una scadenza stretta, oppure il caso è ambiguo e va approfondito. |

Non promettiamo ancora una frequenza fissa come 10 Hz: la latenza reale comprende cattura, trasferimento JPEG, prefill, lettura logprob, azione e verifica. Misurare p50/p95 end-to-end e deadline miss per ambiente. Se la scadenza è superata, l'esecutore applica il comportamento neutro dell'ambiente invece di usare una decisione ormai vecchia. Le soglie per eventuale escalation da SystemOne a chat vanno valutate per il compito, non ricavate direttamente dal campo `confidence`.

Interfaccia minima dell'orchestratore:

```python
observation = sensor.observe()  # timestamp, text_state, optional_jpeg, metadata
if not observation.is_fresh():
    actuator.neutral()
elif task.needs_plan(observation):
    plan = qwen_chat.plan(task.goal, observation.text_state, tools=tools)
else:
    options = actuator.allowed_actions(observation)
    decision = systemone.choose(
        state=task.compact_state(observation),
        image=observation.jpeg,  # None -> nessun frame
        question=task.question,
        options=options,
    )
    actuator.apply(decision, deadline=task.deadline)
task.record(sensor.observe(), decision_or_plan=locals().get("decision", None))
```

È un contratto illustrativo, non codice oggi eseguibile. Lo stato include `goal`, ultime osservazioni rilevanti, posizione/cursore se disponibile, risultato dell'ultima azione e azioni legali. La memoria a lungo termine resta nell'orchestratore; il gateway mantiene soltanto l'ultimo frame per sessione.

## 3. Gaming e benchmark

**Tool da costruire:** `game.observe()` (frame, HUD/telemetria, timestamp), `game.actions()` (azioni legali), `game.apply(action, hold_ms)` (durata limitata), `game.reset(seed)`, `game.result()` (reward, esito, tempo), `game.release_all()`. Se il gioco fornisce API di stato, usarle accanto al frame; per giochi generici il tool di input deve girare sulla macchina che ha il focus del gioco.

**Ciclo:** pianificazione iniziale con Qwen chat; ogni passo acquisisce un'osservazione, pone una domanda corta a SystemOne (es. `move`, `wait`, `release`), esegue per un intervallo definito, poi osserva l'effetto. Qwen chat torna quando il percorso si blocca, l'obiettivo cambia o servono tool nuovi. Il controller decide la durata e impedisce che una vecchia risposta tenga premuti i tasti.

**Benchmark:** partire con [MiniGrid](https://github.com/Farama-Foundation/Minigrid), che ha missioni e azioni discrete; poi una prova visiva 3D come [MineRL](https://github.com/minerllabs/minerl) oppure un gioco locale controllato da screenshot. Registrare seed, numero passi, esito, reward, p50/p95 della decisione, chiamate chat, errori e replay. Confrontare tre condizioni sugli stessi task: sola chat, solo decisioni finite, router ibrido. MiniGrid e MineRL espongono osservazioni/azioni attraverso un'API ambiente, adatta a misure ripetibili.

## 4. Computer-use e Jarvis

**Tool da costruire sul PC Windows:** `desktop.list_windows`, `desktop.inspect_ui(window)` (albero UI Automation), `desktop.screenshot(window_or_region)`, `desktop.cursor_position`, `desktop.click/drag/type/press`, `desktop.wait_for_change`. Un unico esecutore deve serializzare l'input al desktop, che ha un solo focus e cursore. Usare [UI Automation](https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-uiautomationoverview) per controlli con identificatori, screenshot per superfici grafiche e [SendInput](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput) per gesti su canvas. Per pagine web usare locatori e screenshot di [Playwright](https://playwright.dev/docs/locators), anziché coordinate quando esiste un elemento accessibile.

**Esempio Paint:** l'utente chiede «aggiungi un rettangolo accanto al cursore». Il tool restituisce posizione corrente del cursore, finestra/canvas e screenshot. Qwen chat interpreta l'istruzione una volta; SystemOne può scegliere forma/stile o se il canvas è pronto. Le coordinate finali derivano dal cursore e dai limiti della finestra con una funzione deterministica. L'esecutore seleziona lo strumento, trascina, rilascia il mouse e verifica con una nuova immagine. Così «accanto al cursore» usa una coordinata osservata, non una posizione inventata dal modello. Registrare DPI, monitor e rettangolo della finestra insieme allo screenshot.

**Benchmark:** task locali riproducibili prima delle applicazioni personali; [OSWorld V2](https://github.com/xlang-ai/OSWorld-V2) è un riferimento per compiti lunghi di computer-use, con release/task/asset da fissare per confronti corretti.

## 5. Agente autonomo ad eventi

**Tool da costruire:** `events.emit/subscribe` per webhook, file, timer e notifiche locali; `memory.read/write`; `jobs.enqueue/status/cancel`; catalogo dei tool applicativi; `agent.pause/resume/stop`; log di decisioni e risultati. Un evento diventa uno snapshot compatto dello stato, SystemOne decide `ignore | notify | act | ask_model`, Qwen chat pianifica quando serve, l'esecutore richiama tool mirati, e un evento di completamento verifica il risultato.

Per una prima versione singola macchina bastano SQLite per eventi/job e uno scheduler persistente per i timer; [APScheduler](https://apscheduler.readthedocs.io/en/master/userguide.html) distingue trigger, task, job e data store persistente. Se più processi devono condividere code e replay, valutare NATS JetStream; per workflow lunghi con ripresa dopo crash, Temporal dispone di timer, signal e storia riproducibile. Sono opzioni successive, non dipendenze da installare tutte subito.

Un agente continuo necessita idempotency key per evento/azione, deadline, limiti di concorrenza, stato persistente e un interruttore pausa. Le scritture esterne o irreversibili passano attraverso la policy dell'orchestratore. Il modello non deve decidere da solo quali processi sono in esecuzione: i tool descrivono l'ambiente ad ogni ciclo.

## 6. Altri usi che riutilizzano gli stessi componenti

| Caso | Osservazione | Decisione rapida | Tool aggiuntivo |
| --- | --- | --- | --- |
| Triage notifiche e ticket | Testo evento, contesto cliente | `ignore / reply_draft / escalate` | Connettore alla sorgente, invio separato. |
| QA visivo di app/game | Screenshot + stato test | `pass / retry / inspect` | Screenshot, test runner, confronto esito. |
| Sorveglianza di processi locali | Metriche/log, eventuale immagine | `healthy / investigate / restart_candidate` | Metric collector e diagnostica. |
| Assistente contestuale | App in focus, selezione/cursore | `suggest / explain / do_nothing` | UI Automation e memoria della sessione. |
| Simulazione/robotica virtuale | Sensori e frame | Azioni discrete sotto scadenza | Simulatore con `reset/step/reward`. |

Triage e QA senza azioni sul desktop sono i primi casi utili per validare router ed eventi prima di passare al controllo continuo di giochi e applicazioni.

## 7. Sequenza di implementazione

1. **Stabilizzare l'API decisionale:** gestire l'HTTP 500 dei logprob nella build pinned, definire comportamento in errore, testare solo testo e immagine dopo il riavvio del gateway. Nessun aggiornamento automatico di Eugr.
2. **Contratto unico di osservazione/azione:** schema di `Observation`, `Action`, `Result`, timestamp, deadline e log; adapter `:8088` e `:8000`, tool dichiarati da un harness o MCP. MCP standardizza la chiamata dei tool, ma non fornisce da sé screenshot o input: dobbiamo implementarli/esporli dove gira l'ambiente.
3. **Benchmark MiniGrid:** uno strumento `step/reset`, router fast/slow, replay e metriche. Prima di parlare di 10 Hz misurare end-to-end.
4. **Bridge Windows + Paint:** UIA, screenshot, cursore, gesto, verifica; in seguito web/Playwright e giochi reali.
5. **Event loop persistente:** code/scheduler/memoria e integrazioni Jarvis. Migrare a broker/workflow distribuiti solo se il carico o la durata dei task lo richiedono.

Portabilità Qwen4: protocollo di decisione, orchestratore e tool sono indipendenti dal checkpoint. Verificare `chat_template`, visione, token dei label, `logprob_token_ids` e comportamento dei logprob nel nuovo vLLM. Gli hook `hidden_state.py`/`patch_b12x.py` sono legati alla build corrente e non vanno considerati già portati; non sono necessari al percorso decisionale basato sui logprob.
