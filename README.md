COBOL Agentic Analyzer
Un sistema di Agentic AI progettato per automatizzare l'analisi tecnica e il reverse engineering di sistemi legacy COBOL. Il programma trasforma codice sorgente complesso in documentazione strutturata e diagrammi di flusso visivi, facilitando la comprensione di logiche di business datate.

🤖 Come funziona
Il sistema utilizza un'architettura Multi-Agente basata sul pattern Supervisor:
Supervisor Agent: Coordina il flusso di lavoro e smista i task.
Analista Funzionale: Estrae i requisiti e la logica applicativa.
Software Manager: Censimento dei componenti e analisi tecnica.
Diagramma Agent: Genera automaticamente diagrammi di flusso in formato Mermaid.

🛠️ Stack Tecnologico
Linguaggio: Python con interfaccia reattiva Marimo.
Orchestrazione: LangGraph & LangChain.
LLM: Mistral AI (mistral-small-latest).
Protocollo: MCP (Model Context Protocol) per l'accesso sicuro ai file locali.
Monitoring: MLflow per il tracking delle esecuzioni.

📂 Struttura Essenziale
cobol_agent.py: Logica degli agenti e dell'applicazione.
server_mcp.py: Server per l'ingestion dinamica dei file.
/samples: Codice COBOL di test (sorgenti tratti dal progetto open source Accounting-System).

🚀 Setup Rapido
Installazione: pip install -r requirements.txt
API Key: Configura MISTRAL_API_KEY nel tuo ambiente.
Avvio: Esegui marimo edit cobol_agent.py per aprire l'interfaccia di analisi.
