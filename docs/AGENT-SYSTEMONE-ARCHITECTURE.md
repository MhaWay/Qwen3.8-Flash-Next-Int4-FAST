# Endpoint SystemOne interoperabile per harness esterni

Stato: progetto architetturale per `feature/visual-systemone`, 2026-09-29. **Obiettivo del fork: offrire API di decisione e generazione utilizzabili da software esterno. Non costruire nel fork un nuovo harness Minecraft, un controller desktop o Jarvis.** Non introduce un secondo modello né modifica il serving Eugr/B12X. Le componenti chiamate *da costruire* nelle sezioni seguenti sono requisiti dei client esterni, non funzionalità da incorporare nel server.

## 1. Confine tra endpoint e client esterno

```text
Sorgente evento / osservazione
    -> Orchestratore esterno: obiettivo, stato, azioni ammesse, scadenza
        -> :8000/v1/systemone per una scelta tipizzata
        -> :8000/v1/chat/completions per pianificazione e testo
    -> Esecutore locale (gioco, desktop, browser, app)
    -> Nuova osservazione e verifica esito
    -> Log, memoria persistente e trigger successivo
```

Il DGX Spark ospita l'unica istanza Qwen. I programmi esterni osservano e agiscono dove gira l'ambiente: per esempio Mineflayer sul server Minecraft oppure un futuro Jarvis sul PC Windows. **Target: un'unica origine `:8000` con `/v1/systemone`, `/v1/vision/{state_id}/frame` e le route vLLM già esistenti.** L'orchestrazione è responsabilità del programma che le chiama. La porta `:8088` è l'implementazione sperimentale attuale, non un requisito del protocollo.

Per avere davvero la stessa porta, la route SystemOne va installata nell'app FastAPI della build vLLM pinned, come avviene già per la route diagnostica hidden-state, mantenendo un solo ciclo di vita per client HTTP/frame store. In alternativa un reverse proxy potrebbe esporre la stessa origine spostando vLLM su una porta interna, ma non è il percorso preferito per questa build. Non basta impostare `VISUAL_PORT=8000`: la porta è già occupata. Non assumere che i plugin documentati dall'ultima vLLM esistano nella build Eugr pinned; verificare le capacità di quella build prima di implementare. L'integrazione richiede il normale riavvio del server per caricare la patch, da pianificare solo dopo test e review.

Con `state_id` omesso o `null`, `/v1/systemone` non legge un frame e invia a Qwen lo **stato testuale** costruito dai tool, dall'utente o da un precedente passo di Qwen. Non sopprime la richiesta: salta solo la parte visiva. Con `state_id` presente, il frame JPEG corrente della sessione è obbligatorio. Il gateway non osserva da sé lo schermo né richiama strumenti autonomamente.

### Inventario verificato nel fork

| Disponibile ora | Funzione |
| --- | --- |
| Qwen vLLM `:8000/v1/chat/completions` | Generazione, visione e tool calling se un harness fornisce e gestisce realmente i tool. |
| Gateway sperimentale `:8088/v1/systemone` | `state` + domande `choice`, `noul`, `score`; immagine tramite `state_id` oppure solo testo senza `state_id`. Una chiamata vLLM per domanda. Va portato sulla `:8000`. |
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

**Integrazione prevista: usare un progetto Minecraft esistente.** `nthclrd/jevcraft` combina Mineflayer, Jev per la tattica e un LLM OpenAI-compatible per la strategia. `akash-kamat/jev-craft` manda quattro domande nel ciclo reattivo e offre fino a 13 obiettivi nel ciclo tattico. `ellistev/typesafe-minecraft-demo` presenta fino a 20 azioni in una domanda. `teknium1/hermes-and-jev-play-minecraft` usa il formato OpenRouter `/api/alpha/decisions`, che richiede un adapter distinto dal nostro `/v1/systemone`. Mindcraft usa un LLM OpenAI-compatible, ma non risulta un client SystemOne nativo. Tutte queste compatibilità vanno provate sui payload reali, non dedotte dalla somiglianza del nome API. Riferimenti: [jevcraft](https://github.com/nthclrd/jevcraft), [jev-craft](https://github.com/akash-kamat/jev-craft), [demo Minecraft](https://github.com/ellistev/typesafe-minecraft-demo), [Hermes/Jev](https://github.com/teknium1/hermes-and-jev-play-minecraft), [Mindcraft](https://github.com/mindcraft-bots/mindcraft).

Il gateway attuale accetta 4 domande, massimo 8 alternative per `choice/score`, `state` solo testuale di 2048 caratteri e un frame separato per `state_id`. Pertanto **nessuno dei controller sopra è certificato plug-and-play**. Il bersaglio di compatibilità più vicino è `jevcraft` per la doppia API Qwen/Jev; il suo client Jev va verificato per URL configurabile e serializzazione. Un profilo di interoperabilità dovrebbe accettare `state` strutturato, criteria/instructions nel formato inviato dai client, più domande/opzioni, alias di modello `jev-latest` e risposte con forme esatte. Misurare latenza con quattro domande: oggi sono quattro chiamate vLLM distinte. La differenza di porta sparirà con l'integrazione, ma l'eventuale differenza di schema resta da risolvere.

**Tool dei client esterni, già presenti in forma specifica nei progetti citati:** osservazione del mondo, elenco delle azioni legali, esecuzione e verifica. Per Minecraft Mineflayer usa lo stato strutturato del gioco e il protocollo del bot; screenshot e pulsanti di tastiera non sono prerequisiti dell'endpoint.

**Ciclo:** pianificazione iniziale con Qwen chat; ogni passo acquisisce un'osservazione, pone una domanda corta a SystemOne (es. `move`, `wait`, `release`), esegue per un intervallo definito, poi osserva l'effetto. Qwen chat torna quando il percorso si blocca, l'obiettivo cambia o servono tool nuovi. Il controller decide la durata e impedisce che una vecchia risposta tenga premuti i tasti.

**Benchmark:** prima collegare uno dei controller Minecraft esistenti e registrare seed, numero passi, esito, p50/p95 della decisione, chiamate chat ed errori. Confrontare tre configurazioni dello stesso client: LLM solo, decisioni tipizzate, combinazione. [MiniGrid](https://github.com/Farama-Foundation/Minigrid) e [MineRL](https://github.com/minerllabs/minerl) restano test aggiuntivi se vogliamo isolare la qualità del motore decisionale dal comportamento di Mineflayer.

## 4. Computer-use e Jarvis

**Progetto esterno.** Questa sezione descrive i requisiti che un Jarvis separato avrebbe per chiamare il nostro endpoint; non è un piano per implementare screenshot e input nel fork del modello.

**Tool da costruire sul PC Windows:** `desktop.list_windows`, `desktop.inspect_ui(window)` (albero UI Automation), `desktop.screenshot(window_or_region)`, `desktop.cursor_position`, `desktop.click/drag/type/press`, `desktop.wait_for_change`. Un unico esecutore deve serializzare l'input al desktop, che ha un solo focus e cursore. Usare [UI Automation](https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-uiautomationoverview) per controlli con identificatori, screenshot per superfici grafiche e [SendInput](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput) per gesti su canvas. Per pagine web usare locatori e screenshot di [Playwright](https://playwright.dev/docs/locators), anziché coordinate quando esiste un elemento accessibile.

**Esempio Paint:** l'utente chiede «aggiungi un rettangolo accanto al cursore». Il tool restituisce posizione corrente del cursore, finestra/canvas e screenshot. Qwen chat interpreta l'istruzione una volta; SystemOne può scegliere forma/stile o se il canvas è pronto. Le coordinate finali derivano dal cursore e dai limiti della finestra con una funzione deterministica. L'esecutore seleziona lo strumento, trascina, rilascia il mouse e verifica con una nuova immagine. Così «accanto al cursore» usa una coordinata osservata, non una posizione inventata dal modello. Registrare DPI, monitor e rettangolo della finestra insieme allo screenshot.

**Benchmark:** task locali riproducibili prima delle applicazioni personali; [OSWorld V2](https://github.com/xlang-ai/OSWorld-V2) è un riferimento per compiti lunghi di computer-use, con release/task/asset da fissare per confronti corretti.

## 5. Agente autonomo ad eventi

**Progetto esterno.** La persistenza, i trigger e le azioni vivono nell'applicazione autonoma; l'endpoint risponde a richieste, non mantiene un ciclo autonomo.

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

## 7. Sequenza per il nostro fork, poi esperimenti esterni

1. **Registrare i payload dei client esistenti** per Minecraft, browser e SDK: forme di `state`, domande, modelli, autenticazione, URL e risposte attese. Non cambiare l'API sulla base di esempi inventati.
2. **Stabilizzare il gateway pinned:** gestire l'HTTP 500 dei logprob, comportamento in errore, testare solo testo e immagine in isolamento. Nessun aggiornamento automatico di Eugr.
3. **Ampliare il profilo di compatibilità** dove i test lo giustificano: `state` JSON, criteria strutturati, opzioni e domande in numero sufficiente, alias/modello, eventuale auth opzionale. Esporre un documento di capacità/versione. Evitare di dichiarare parità con la calibrazione Jev senza misurarla.
4. **Montare le route nello stesso server vLLM sulla `:8000`**, con test di avvio/health, chat, decisioni testuali e visive, store e spegnimento. Mantenere la `:8088` soltanto come modalità di prova fino alla migrazione.
5. **Provare il client Minecraft più vicino**, usando la stessa origine `:8000` per decisioni e Qwen chat; limitare l'adapter ai confini HTTP/schema del progetto esterno. Misurare errori, latenza e decisioni per secondo. Mindcraft resta un test della sola API chat finché non gli si aggiunge un client SystemOne.
6. **Solo in seguito** costruire Jarvis/desktop e altri programmi come repository separati che consumano queste API. MiniGrid può restare un benchmark supplementare, non la prima integrazione richiesta.

Portabilità Qwen4: protocollo di decisione, orchestratore e tool sono indipendenti dal checkpoint. Verificare `chat_template`, visione, token dei label, `logprob_token_ids` e comportamento dei logprob nel nuovo vLLM. Gli hook `hidden_state.py`/`patch_b12x.py` sono legati alla build corrente e non vanno considerati già portati; non sono necessari al percorso decisionale basato sui logprob.
