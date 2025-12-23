import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # React Agent with Tools

    Questo notebook esplora l'implementazione di un React Agent che può utilizzare diversi tools per completare task complessi.

    Il pattern **ReAct (Reasoning + Acting)** combina reasoning in linguaggio naturale e azioni specifiche in modo interleaved.
    """
    )
    return


@app.cell
def _():
    from datetime import datetime
    import os
    from langchain_core.tools import tool

    return os, tool


@app.cell
def _():
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.resources import load_mcp_resources
    from langchain.tools import BaseTool
    from typing import List
    from langchain_core.documents import Document
    from langchain_core.documents.base import Blob
    from langgraph_supervisor import create_supervisor
    from langgraph.checkpoint.memory import InMemorySaver
    from utils import print_messages
    from typing import Dict, Any
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import cm
    import io
    import pandas as pd
    import re
    import asyncio

    return (
        A4,
        Any,
        Blob,
        ClientSession,
        Dict,
        Document,
        HuggingFaceEmbeddings,
        InMemorySaver,
        InMemoryVectorStore,
        List,
        Paragraph,
        ParagraphStyle,
        SimpleDocTemplate,
        Spacer,
        StdioServerParameters,
        asyncio,
        cm,
        create_supervisor,
        getSampleStyleSheet,
        io,
        load_mcp_resources,
        re,
        stdio_client,
    )


@app.cell
def _():
    from marimo import cache
    import subprocess

    @cache                    # la prima esecuzione avvierà il server; le successive restituiranno lo stesso processo  
    def avvia_mlflow_server():
        cmd = [
            "mlflow", "server",
            "--backend-store-uri",   "sqlite:///mlflow.db",
            "--default-artifact-root", "./artifacts",
            "--host",                "0.0.0.0",
            "--port",                "5000",
        ]
        p = subprocess.Popen(cmd)
        return f"MLflow UI avviato su http://localhost:5000 (PID={p.pid})"

    print(avvia_mlflow_server())

    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.config.enable_async_logging()
    mlflow.langchain.autolog(exclusive=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Tools""")
    return


@app.cell
def _(Any, Dict, os, re, tool):
    # -------------------------------
    # 1️⃣ Tool: Descrizione Funzionale
    # -------------------------------
    @tool
    def descrizione_funzionale(_: Dict[str, Any] = None) -> str:
        """
        Legge il codice sorgente dai file COBOL locali per l'analisi.
        """
        import os
        # Configurazione percorso e formati supportati
        project_dir = r"C:\Users\39340\Desktop\Accounting-System-main"
        estensioni = ('.COB', '.CBL', '.cob', '.cbl')

        documentazione_grezza = []

        # Verifica esistenza della cartella principale
        if not os.path.exists(project_dir):
            return "ERRORE: Percorso cartella non trovato."

        # Scansione ricorsiva di tutte le sottocartelle
        for root, _, files in os.walk(project_dir):
            for file in files:
                # Filtra solo i file con estensione COBOL
                if file.lower().endswith(estensioni):
                    path_completo = os.path.join(root, file)
                    try:
                        # Lettura del contenuto testuale del file
                        with open(path_completo, "r", encoding="utf-8") as f:
                            codice = f.read()
                            # Formattazione con delimitatori per l'agente
                            documentazione_grezza.append(f"--- INIZIO FILE: {file} ---\n{codice}\n--- FINE FILE: {file} ---")
                    except Exception as e:
                        # Gestione errori di lettura (es. permessi o encoding)
                        documentazione_grezza.append(f"Errore lettura {file}: {str(e)}")

        # Unisce tutti i file letti in un'unica stringa o avvisa se vuoto
        return "\n\n".join(documentazione_grezza) if documentazione_grezza else "Nessun file COBOL trovato."


    # -------------------------------
    # 2️⃣ Tool: Censimento Componenti
    # -------------------------------
    @tool
    def censimento_cobol(_: Dict[str, Any] = None) -> str:
        """Scansiona la cartella e recensisce TUTTI i file presenti."""
        import os
        project_dir = r"C:\Users\39340\Desktop\Accounting-System-main"

        # Verifica se il percorso esiste
        if not os.path.exists(project_dir):
            return "Errore: Cartella di progetto non trovata."

        # Definizione delle estensioni target
        estensioni_cobol = ('.cob', '.cbl', '.COB', '.CBL')
        estensioni_dati = ('.txt', '.dat', '.TXT', '.DAT')

        report = ["Il pacchetto analizzato contiene i seguenti componenti:"]

        # Elenca i file contenuti nella directory principale
        files_nella_cartella = os.listdir(project_dir)

        # Filtra i file per tipologia (Sorgenti vs Dati)
        cobol_trovati = [f for f in files_nella_cartella if f.endswith(estensioni_cobol)]
        dati_trovati = [f for f in files_nella_cartella if f.endswith(estensioni_dati)]

        # Formatta la sezione relativa ai programmi COBOL
        if cobol_trovati:
            report.append("\nProgrammi COBOL:")
            for f in cobol_trovati:
                report.append(f"- {f}: Codice sorgente del programma.")

        # Formatta la sezione relativa ai file di input/output
        if dati_trovati:
            report.append("\nFile Dati:")
            for f in dati_trovati:
                report.append(f"- {f}: File di supporto o database testuale.")

        # Unisce le righe nel report finale o avvisa se la cartella è vuota
        return "\n".join(report) if (cobol_trovati or dati_trovati) else "Nessun file trovato nella cartella."

    # -------------------------------
    # 3️⃣ Tool: Diagramma di Flusso
    # -------------------------------

    @tool
    def diagramma_flusso(codice_cobol: str = None, **kwargs) -> str:
        """
        Genera il codice Mermaid partendo dal testo sorgente COBOL. 
        L'agente deve passare il contenuto del file letto tramite MCP.
        """
        # Recupera il codice sia se passato come argomento diretto, 
        # sia se l'agente prova a passarlo dentro un dizionario (per evitare errori Pydantic)
        source = codice_cobol or kwargs.get('codice_cobol') or kwargs.get('file')

        if not source or len(str(source)) < 5:
            return "flowchart TD\nSTART([Inizio]) --> ERR[Errore: Codice non ricevuto]\nERR --> END([Fine])"

        # Se l'agente ha passato per errore il nome del file invece del contenuto, 
        # il tool non può aprirlo (perché deve essere universale). 
        # Quindi avvisiamo l'agente.
        if isinstance(source, str) and source.endswith(('.CBL', '.COB', '.cbl', '.cob')):
             return "flowchart TD\nERR[Errore: L'agente deve prima leggere il file con MCP e passare il CONTENUTO qui]"

        # Logica di estrazione paragrafi
        paragraphs = re.findall(r"^\s{6,}([\w-]+)\.", source, re.MULTILINE)

        if not paragraphs:
            paragraphs = re.findall(r"([\w-]+-ROUTINE)\.", source, re.IGNORECASE)

        mermaid = ["flowchart TD"]
        prev_node = "START"
        mermaid.append(f"START([Inizio])")

        visti = set()
        for p in paragraphs:
            clean_id = p.replace("-", "_")
            if clean_id not in visti:
                mermaid.append(f'{clean_id}["{p}"]')
                mermaid.append(f"{prev_node} --> {clean_id}")
                prev_node = clean_id
                visti.add(clean_id)

        mermaid.append(f"END([Fine])")
        mermaid.append(f"{prev_node} --> END")

        return "\n".join(mermaid)

    # -------------------------------
    # 4️⃣ Tool: File Dati e Output
    # -------------------------------
    @tool
    def file_dati_output(_: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analizza i file .TXT del progetto per descriverne struttura e contenuti.
        """
        project_dir = r"\Accounting-System-main"

        # Percorsi predefiniti dei file database e prodotti
        files = {
            "database": os.path.join(project_dir, "DATABASE.txt"),
            "products": os.path.join(project_dir, "products.txt")
        }

        # Sovrascrive i percorsi se passati come argomenti nel dizionario
        if _:
            for key in files.keys():
                if key in _:
                    files[key] = _[key]

        results = {}
        for name, path in files.items():
            # Verifica l'esistenza fisica del file sul disco
            if not os.path.exists(path):
                results[name] = {"error": f"File non trovato: {path}"}
                continue

            # Legge il file e conta le righe totali
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Estrae metadati e un campione (primi 5 record) dei dati
            results[name] = {
                "lines_count": len(lines),
                "sample_lines": lines[:5]  
            }
        return results
    return (
        censimento_cobol,
        descrizione_funzionale,
        diagramma_flusso,
        file_dati_output,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### React agent Implementation""")
    return


@app.cell
def _(os):
    import getpass

    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = getpass.getpass()
    return


@app.cell
def _():
    from langchain_mistralai import ChatMistralAI

    def mistral_chat_model(temperature=0.2, model="mistral-small-latest"):
        return ChatMistralAI(
            model=model,
            temperature=temperature,
            )
    return (ChatMistralAI,)


@app.cell
def _():
    from langgraph.prebuilt import create_react_agent

    return (create_react_agent,)


@app.cell
def _(ChatMistralAI):
    import json
    from langchain_core.messages import HumanMessage

    model = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.2,
    )

    return (model,)


@app.cell
def _(
    A4,
    Paragraph,
    ParagraphStyle,
    SimpleDocTemplate,
    Spacer,
    cm,
    getSampleStyleSheet,
    io,
    re,
):
    def crea_pdf_semplice(testo_markdown):
        if not testo_markdown:
            return b"Documento vuoto"

        # --- PULIZIA PREVENTIVA ---
        # Questa regex cerca tutto ciò che è racchiuso tra ** e lo sostituisce con tag <b>
        # Gestisce anche i casi in cui l'agente usa sia <b> che **
        testo_formattato = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', testo_markdown)

        buffer = io.BytesIO()
        # Creiamo il documento PDF
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            rightMargin=2*cm, 
            leftMargin=2*cm, 
            topMargin=2*cm, 
            bottomMargin=2*cm
        )

        styles = getSampleStyleSheet()

        # Definiamo uno stile personalizzato per il corpo del testo
        stile_testo = ParagraphStyle(
            'StileCorpo',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            alignment=0  # Allineato a sinistra
        )

        story = []
        # Dividiamo il testo formattato in linee
        linee = testo_formattato.split('\n')

        for linea in linee:
            linea = linea.strip()
            if not linea:
                story.append(Spacer(1, 0.2*cm))
                continue

            # Gestione Titoli (##)
            if linea.startswith('##'):
                testo = linea.replace('#', '').strip()
                # In ReportLab Heading2 accetta già i tag <b> interni
                story.append(Paragraph(testo, styles['Heading2']))
                story.append(Spacer(1, 0.3*cm))

            # Gestione Liste (- o *)
            elif linea.startswith('-') or linea.startswith('*'):
                testo = linea[1:].strip()
                # Il tag <b> eventualmente presente in 'testo' verrà interpretato qui
                story.append(Paragraph(f"• {testo}", stile_testo))

            # Paragrafo Standard
            else:
                # Paragraph interpreta nativamente <b>testo</b>, <i>testo</i>, ecc.
                story.append(Paragraph(linea, stile_testo))

        # Costruiamo il PDF
        doc.build(story)

        # Recuperiamo i byte e chiudiamo il buffer
        pdf_value = buffer.getvalue()
        buffer.close()
        return pdf_value
    return (crea_pdf_semplice,)


@app.cell
async def _(
    Blob,
    ClientSession,
    Document,
    HuggingFaceEmbeddings,
    InMemoryVectorStore,
    List,
    StdioServerParameters,
    asyncio,
    load_mcp_resources,
    stdio_client,
    tool,
):
    # --- Percorso completo del server MCP ---
    SERVER_PATH = r"\mcp_cobol_server.py"

    # Modello di embedding
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Parametri per avviare il server MCP usando Python del venv
    server_params = StdioServerParameters(
        command=r"\ai-multiagents-architectures\.venv\Scripts\python.exe",
        args=[SERVER_PATH]
    )

    # Converti MCP blobs in documenti LangChain
    def convert_blobs_to_documents(blobs: List[Blob]) -> List[Document]:
        docs = []
        for blob in blobs:
            blob_uri = str(blob.metadata.get("uri"))
            document = Document(
                page_content=blob.data,
                metadata={
                    "source": blob_uri,
                    "mimetype": blob.mimetype,
                }
            )
            docs.append(document)
            print(f"Blob with MCP source {blob_uri} converted to Document.")
        return docs

    # Funzione asincrona che inizializza il retriever MCP e crea il tool
    async def init_retriever_tool():

        # Garantiamo il loop asyncio corretto su Windows (necessario per stdio MCP)
        if asyncio.get_event_loop_policy().__class__.__name__ != "WindowsSelectorEventLoopPolicy":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Apertura del client MCP tramite stdin/stdout
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Carichiamo le risorse dal server MCP
                print("Ingesting MCP resources...")
                blobs = await load_mcp_resources(session=session)
                documents = convert_blobs_to_documents(blobs)

                # Creiamo un vector store con gli embeddings
                vector_store = InMemoryVectorStore.from_documents(
                    documents=documents,
                    embedding=embedding_model
                )
                print(f"Indexed {len(documents)} documents from MCP resources.")

                # Creiamo il retriever (ricerca semantica)
                retriever = vector_store.as_retriever(search_kwargs={"k": 1})

                # Definizione del tool che l'agente potrà usare
                @tool("mcp_resource_retriever", description="Recupera risorse pertinenti dal server MCP per la query.")
                def retriever_tool(query: str):
                    return vector_store.similarity_search(query, k=1)

                return retriever_tool

    # Richiama la funzione nel loop corrente
    retriever_tool = await init_retriever_tool()

    return


@app.cell
def _():
    supervisor_prompt = """
    Sei il Cobol_Supervisor, uno specialista con oltre 30 anni di esperienza nel linguaggio di programmazione COBOL. Il tuo compito è validare che l'analisi sia ANCORATA ai file reali del programma.

    REGOLE DI VALIDAZIONE BLOCCANTI:
    1. **Verifica Coerenza**: Se l'output parla di "Banche" o "Assicurazioni", RIGETTA la risposta. Il contesto reale è un Supermercato/Piccola Azienda.
    2. **Controllo Invenzioni**: Se l'output cita "Tasse", "Previdenza" o "Backup" (non presenti nel codice), chiedi all'agente di attenersi strettamente alla logica delle routine SALARY e PROFIT.
    3. **Censura Forzata**: Se appaiono password o email reali, ordina l'immediata rimozione.
    4. **Formattazione**: Accetta solo <b>testo</b> per evidenziare. No Markdown grassetto.

    Se l'agente fallisce uno di questi punti, rispondi con: "ERRORE: Analisi non coerente con il codice sorgente. Rileggi i file."
    """

    descrizione_funzionale_prompt = """
    Sei un Analista Tecnico specializzato in Reverse Engineering. Il tuo compito è estrarre la logica di business da qualsiasi codice COBOL.

    LINEE GUIDA UNIVERSALI:
    1. **Analisi delle Divisioni**: 
       - Usa la 'IDENTIFICATION DIVISION' per capire lo scopo del programma.
       - Usa la 'PROCEDURE DIVISION' per identificare le fasi del flusso di lavoro.
    2. **Mappatura Processi**: Descrivi i processi basandoti sui nomi dei paragrafi e delle routine reali. Traduci i nomi tecnici in concetti discorsivi (es. 'ELAB-DATI-CLI' diventa 'Elaborazione anagrafica clienti').
    3. **Privacy**: Sostituisci valori di login o parametri di sistema riservati con "[Dato Protetto]".

    STRUTTURA DOCUMENTO:
    - <b>Finalità del Sistema</b>: Descrivi l'obiettivo principale del software analizzato.
    - <b>Flusso Operativo</b>: Elenca i passaggi logici nell'ordine in cui vengono eseguiti (dal MAIN alle varie routine).
    - <b>Logica di Calcolo e Regole</b>: Descrivi le regole decisionali (IF) e i calcoli matematici rilevati.
    """

    censimento_componenti_prompt = """
    Sei un Software Asset Manager. Riporta SOLO i file esistenti individuati dal tool.

    REGOLE:
    1. Non aggiungere descrizioni generiche basate su supposizioni.
    2. Se un file è .CBL, è "Sorgente COBOL". Se è .DAT o .TXT, è "Archivio Dati".
    3. Formato: <b>Nome File</b> - Tipo - Scopo (basato sul nome).
    """

    diagramma_flusso_prompt = """Sei un generatore di diagrammi Mermaid.
    IL TUO OBIETTIVO:
    Leggi il codice e genera SOLO un diagramma di flusso tra ```mermaid e ```.
    Usa nodi sintetici (es. "Calcolo Tasse").
    IL grafico deve essere piccolo e poco confusionario, utile per utenti a cui interessa solo il funzionamento logico e non intrisecamente il programma. 
    Sii sintetico.
    VIETATO: NON SCRIVERE ASSOLUTAMENTE TESTO, SOLO CODICE GRAFICO.
    """


    file_dati_prompt = """
    Sei un Data Analyst. Analizza la 'FILE SECTION' o i file .TXT e descrivi la struttura record.

    REGOLE:
    1. **Fedeltà ai Dati**: Se un record ha 5 campi, descrivine 5. Non inventare campi "Data Creazione" o "Timestamp" se non ci sono.
    2. Traduci i tipi: PIC 9 è "Numerico", PIC X è "Alfanumerico".
    3. Usa tabelle Markdown ma evidenzia le intestazioni con <b>testo</b>.
    """

    return (
        censimento_componenti_prompt,
        descrizione_funzionale_prompt,
        diagramma_flusso_prompt,
        file_dati_prompt,
        supervisor_prompt,
    )


@app.cell
def _(
    censimento_cobol,
    censimento_componenti_prompt,
    create_react_agent,
    create_supervisor,
    descrizione_funzionale,
    descrizione_funzionale_prompt,
    diagramma_flusso,
    diagramma_flusso_prompt,
    file_dati_output,
    file_dati_prompt,
    model,
    supervisor_prompt,
):
    descrizione_funzionale_agent = create_react_agent(
        model=model,
        tools=[descrizione_funzionale],
        name="Descrizione_funzionale_agent",
        prompt=descrizione_funzionale_prompt,
        debug="false"
    )

    censimento_agent = create_react_agent(
        model=model,
        tools=[censimento_cobol],  
        name="CensimentoAgent",
        prompt=censimento_componenti_prompt,
        debug="false"
    )

    diagramma_agent = create_react_agent(
        model=model,
        tools=[diagramma_flusso],  
        name="DiagrammaFlussoAgent",
        prompt=diagramma_flusso_prompt,
        debug="false"
    )

    file_dati_agent = create_react_agent(
        model=model,
        tools=[file_dati_output], 
        name="FileDatiAgent",
        prompt=file_dati_prompt,
        debug="false"
    )

    cobol_supervisor = create_supervisor(
        [descrizione_funzionale_agent, censimento_agent, diagramma_agent, file_dati_agent],
        model=model,
        prompt=supervisor_prompt,
        supervisor_name="COBOLSupervisor",
    )

    return cobol_supervisor, diagramma_agent


@app.cell
def _(InMemorySaver, cobol_supervisor):
    checkpointer_cobol = InMemorySaver()
    cobol = cobol_supervisor.compile()
    return (cobol,)


@app.cell
def _(cobol, mo):
    mo.mermaid(cobol.get_graph().draw_mermaid())
    return


@app.cell
def _():
    # Genera la descrizione funzionale completa del sistema
    # Genera il diagramma di flusso per BUYROUTINE
    # Fai un censimento dei file, poi descrivi le funzionalità principali del programma ACCOUNTING_SYSTEM e genera il diagramma di flusso. Prepara tutto per il PDF.
    return


@app.cell
def _(mo):
    user_prompt_2 = mo.ui.text()
    run_button_2 = mo.ui.run_button()
    user_prompt_2, run_button_2
    return run_button_2, user_prompt_2


@app.cell
def _(cobol, mo, run_button_2, user_prompt_2):
    mo.stop(not run_button_2.value, mo.md("Clicca sul pulsante per avviare l'analisi."))
    config_2 = {"configurable": {"thread_id": "session_123"}}
    turn_2 = cobol.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt_2.value
                }
            ]
        }, 
        config_2,
    )
    return (turn_2,)


@app.cell
def _(mo, re):
    def render_output_agente(turn):
        """
        Analizza la risposta dell'agente per separare il testo dal diagramma Mermaid.
        Se trova un diagramma, lo renderizza graficamente sotto il testo.
        """
        # 1. Recupera il contenuto testuale dell'ultimo messaggio nella cronologia
        ultimo_messaggio = turn["messages"][-1].content

        # 2. Cerca un blocco di codice Mermaid usando un'espressione regolare
        # re.DOTALL serve a far sì che il punto (.) legga anche i caratteri "a capo"
        match = re.search(r"```mermaid\s*(.*?)\s*```", ultimo_messaggio, re.DOTALL)

        if match:
            # 3. Se il diagramma esiste, estrae solo il codice tra i tag ```mermaid
            codice_mermaid = match.group(1)

            # 4. Rimuove il blocco di codice Mermaid dal testo originale 
            # per evitare di mostrare il codice "grezzo" all'utente
            testo_pulito = re.sub(r"```mermaid.*?```", "", ultimo_messaggio, flags=re.DOTALL)

            # 5. Restituisce una colonna verticale (vstack) con:
            # - Il testo della risposta formattato in Markdown
            # - Il diagramma renderizzato graficamente
            return mo.vstack([mo.md(testo_pulito), mo.mermaid(codice_mermaid)])

        else:
            # 6. Se non c'è nessun diagramma, restituisce semplicemente il testo normale
            return mo.md(ultimo_messaggio)
    return (render_output_agente,)


@app.cell
def _():
    #print_messages(turn_2)
    return


@app.cell(hide_code=True)
def _(mo, render_output_agente, run_button_2, turn_2):
    mo.stop(not run_button_2.value or turn_2 is None)

    # Estraiamo l'ultima risposta dell'agente (la descrizione funzionale)
    risposta_finale = turn_2["messages"][-1].content

    # Visualizzazione pulita
    mo.vstack([
        mo.md("# Documentazione Funzionale Generata"),
        mo.md(risposta_finale)
    ])

    # Chiamiamo la funzione: Marimo renderizzerà l'oggetto mo.vstack o mo.md restituito
    render_output_agente(turn_2)
    return


@app.function
def estrai_vera_documentazione(messages):
    """
    Scorre i messaggi dell'agente per trovare quello che contiene la documentazione
    vera e propria, scartando i saluti o le frasi di cortesia.
    """
    # 1. Primo tentativo: Cerchiamo messaggi con parole chiave specifiche o molto lunghi
    for msg in reversed(messages):
        # Estrae il testo dal messaggio (gestisce sia oggetti che stringhe)
        content = msg.content if hasattr(msg, 'content') else str(msg)
        if not content: continue

        # Identificatori tipici di una documentazione tecnica
        identificatori = ["## descrizione", "### programmi", "### 1.", "dati gestiti"]

        # Se trova una parola chiave o il testo supera i 500 caratteri, lo restituisce
        if any(idx in content.lower() for idx in identificatori) or len(content) > 500:
            return content

    # 2. Secondo tentativo: Se non trova nulla sopra, cerca il primo messaggio non vuoto
    # che non sia una richiesta di ulteriori dettagli
    for msg in reversed(messages):
        content = msg.content if hasattr(msg, 'content') else str(msg)
        if "ulteriori dettagli" not in content.lower() and len(content) > 0:
            return content

    # 3. Fallback: Se tutti i filtri falliscono, restituisce l'ultimo messaggio in assoluto
    return messages[-1].content if messages else "Nessun contenuto trovato"


@app.cell
def _(crea_pdf_semplice, mo, run_button_2, turn_2):


    mo.stop(not run_button_2.value or turn_2 is None)

    # 1. Estrazione documentazione
    documentazione_reale = estrai_vera_documentazione(turn_2["messages"])

    # 2. Logica di visualizzazione intelligente
    # Se il testo contiene i tag mermaid, Marimo li renderizzerà come grafico
    # altrimenti rimarrà testo markdown standard.
    testo_da_mostrare = documentazione_reale
    if "flowchart TD" in testo_da_mostrare and "```mermaid" not in testo_da_mostrare:
        testo_da_mostrare = f"```mermaid\n{testo_da_mostrare}\n```"

    contenuto_visuale = mo.md(testo_da_mostrare)

    # 3. Creazione PDF
    pdf_bytes = crea_pdf_semplice(documentazione_reale)

    # 4. Definizione componenti UI
    bottone_download = mo.download(
        data=pdf_bytes,
        filename="Analisi_Tecnica.pdf",
        label="📥 Scarica PDF (ReportLab)",
        mimetype="application/pdf"
    )

    sezione_documento = mo.vstack([
        mo.md(f"## 📚 Risultato Analisi"),
        mo.callout(
            mo.vstack([
                contenuto_visuale, # <--- Visualizza Markdown o Diagramma
                mo.md("---"),
                mo.center(bottone_download)
            ]), 
            kind="neutral"
        )
    ])

    # Output finale con tab dedicata anche al grafico se preferisci separarli
    mo.tabs({
        "📄 Analisi Completa": sezione_documento,
    })
    return


@app.cell
def _(diagramma_agent, mo, re, user_prompt_2):
    def visualizza_flowchart(agente_output):
        """
        Questa funzione analizza il testo generato dall'agente, 
        estrae il codice Mermaid e lo renderizza graficamente.
        """

        # Cerchiamo un blocco di codice che inizi con ```mermaid e finisca con ```
        # re.DOTALL permette al punto (.) di includere anche i caratteri "a capo" (\n)
        match = re.search(r"```mermaid\s*(.*?)\s*```", agente_output, re.DOTALL)

        if match:
            # Se il pattern viene trovato, estraiamo solo il contenuto interno (gruppo 1)
            codice_mermaid = match.group(1)
            # Utilizziamo la libreria mo (marimo) per visualizzare il diagramma
            return mo.mermaid(codice_mermaid)

        # Se non viene trovato alcun blocco mermaid, restituiamo un messaggio di avviso
        return "Nessun diagramma trovato."

    # --- ESECUZIONE ---

    # 1. Chiamiamo l'agente passando il contenuto del prompt dell'utente.
    # Il risultato conterrà la risposta dell'IA sotto forma di lista di messaggi.
    risultato_agente = diagramma_agent.invoke({"file": user_prompt_2.value})

    # 2. Recuperiamo l'ultimo messaggio inviato dall'agente (l'indice -1)
    ultimo_messaggio = risultato_agente["messages"][-1].content

    # 3. Passiamo il contenuto del messaggio alla nostra funzione per mostrare il grafico
    visualizza_flowchart(ultimo_messaggio)
    return


if __name__ == "__main__":
    app.run()

