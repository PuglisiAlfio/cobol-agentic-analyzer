COBOL Agentic Analyzer
Un sistema Multi-Agente avanzato basato su architettura Agentic AI per automatizzare l'analisi tecnica e il reverse engineering di codice legacy COBOL. Il progetto utilizza il pattern Supervisor per coordinare agenti specializzati e trasformare programmi complessi in documentazione chiara e diagrammi di flusso dinamici.

🚀 Caratteristiche principali
Architettura Multi-Agente: Utilizzo di langgraph-supervisor per gestire un flusso di lavoro tra agenti specializzati (Analista Funzionale, Software Manager, Diagramma Agent).
Integrazione MCP (Model Context Protocol): Ingestion dinamica di risorse locali tramite un server MCP dedicato per l'analisi dei file system.
Analisi AI ReAct: Implementazione del pattern Reasoning + Acting per estrarre logica di business da sorgenti COBOL.
Output Multi-formato: Generazione automatica di diagrammi di flusso in formato Mermaid e report tecnici esportabili in PDF tramite ReportLab.

🛠️ Stack Tecnologico
Core: Python, Marimo Notebooks.
AI Framework: LangChain, LangGraph.
LLM: Mistral AI (modello mistral-small-latest).
Data Injection: MCP (Model Context Protocol).
Tracking: MLflow per il monitoraggio delle esecuzioni.

📂 Struttura del Progetto
cobol_agent.py: Logica principale del sistema e definizione degli agenti.
server_mcp.py: Implementazione del server per il recupero delle risorse locali.
utils.py: Funzioni di supporto per la gestione dei messaggi e del logging.
/samples: Cartella contenente i file COBOL e i dati di test.

📚 Fonte dei dati
I file COBOL utilizzati per i test e la dimostrazione delle capacità dell'agente appartengono al progetto open source Accounting-System di osha-san.
⚙️ Installazione e Utilizzo
Clona il repository.
Installa le dipendenze: pip install -r requirements.txt.
Imposta la tua chiave API Mistral: export MISTRAL_API_KEY='tua_chiave'.
Avvia l'applicazione con Marimo: marimo edit cobol_agent.py.
