import asyncio
import io

import streamlit as st

from baml_client.async_client import b
from rag import (
    answer_question,
    execute_vector_and_fts_rag,
    prune_schema,
)

st.set_page_config(page_title="Hybrid RAG Interactive Demo", layout="centered")
st.title("Hybrid (graph + vector + FTS) RAG")

question = st.text_input(
    f"Enter your question:",
    f"How many patients have been immunized after January 1, 2022?",
    key=f"question_input",
    help=(f"Type your question about the FHIR dataset here and press Enter or click Run."),
)
run_clicked = st.button("Run", type="primary")

log_buffer = io.StringIO()

# Store cypher query in session state for display
if "cypher_query" not in st.session_state:
    st.session_state["cypher_query"] = None


async def run_pipeline_steps(question, log_buffer, status):
    logs = []
    cypher_query = None

    def log(msg):
        logs.append(msg)
        log_buffer.write(msg + "\n")
        status.code(msg, language="text")

    log("[1/6] Pruning schema...")
    pruned_schema_xml = await prune_schema(question)
    log("[1/6] Pruned schema XML generated.")

    log("[2/6] Extracting entities...")
    entities = await b.ExtractEntityKeywords(question, pruned_schema_xml)
    important_entities = " ".join(
        [f"{entity.key} {entity.value}".replace("_", " ") for entity in entities]
    )
    log(f"[2/6] Extracted entities: {important_entities}")

    log("[3/6] Generating vector/FTS context...")
    vector_context = await execute_vector_and_fts_rag(
        question, pruned_schema_xml, important_entities
    )
    log("[3/6] Vector/FTS context generated.")

    log("[4/6] Generating Cypher and graph answer...")
    cypher_response = await b.Text2Cypher(question, pruned_schema_xml, important_entities)
    if cypher_response.cypher:
        cypher_query = cypher_response.cypher
        st.session_state["cypher_query"] = cypher_query
        log(f"[4/6] Cypher generated.")
        # Run the Cypher query on the graph database
        from utils import KuzuDatabaseManager

        kuzu_db_manager = KuzuDatabaseManager("./fhir_kuzu_db")
        conn = kuzu_db_manager.get_connection()
        query = cypher_query
        response = conn.execute(query)
        result = response.get_as_pl().to_dicts()  # type: ignore
        context = f"<CYPHER>\n{query}\n</CYPHER>\n<RESULT>\n{result}\n</RESULT>"
        log("[4/6] Cypher query executed and result obtained.")
    else:
        st.session_state["cypher_query"] = None
        log("[4/6] No Cypher query was generated.")
        context = ""
    graph_answer = await answer_question(question, context)
    log("[4/6] Graph answer generated.")

    log("[5/6] Generating vector answer...")
    vector_answer = await answer_question(question, vector_context)
    log("[5/6] Vector answer generated.")

    log("[6/6] Synthesizing final answer...")
    synthesized_answer = await b.SynthesizeAnswers(question, vector_answer, graph_answer)
    log("[6/6] Final answer synthesized.")
    return synthesized_answer, "\n".join(logs)


final_answer_container = st.empty()
cypher_query_container = st.empty()

if run_clicked and question.strip():
    # Reset cypher query for new run
    st.session_state["cypher_query"] = None
    with st.status("Running Hybrid RAG pipeline...", expanded=False) as status:
        answer, logs = asyncio.run(run_pipeline_steps(question, log_buffer, status))
        status.update(label="Done!", state="complete")
        st.session_state["logs"] = logs
        st.session_state["final_answer"] = answer

# When we have the final answer
if "final_answer" in st.session_state:
    final_answer_container.subheader("Final Answer:")
    final_answer_container.write(st.session_state["final_answer"])

# When we have the cypher query
if st.session_state.get("cypher_query"):
    with cypher_query_container.expander("Generated Cypher query", expanded=True):
        st.code(st.session_state["cypher_query"], language="cypher")
