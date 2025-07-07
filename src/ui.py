import asyncio

import streamlit as st

from graphrag import run_graphrag

st.set_page_config(page_title="Graph RAG Chat", layout="wide")
st.title("Graph RAG Interactive Chat")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of dicts: {question, cypher, context, answer}

# User input
with st.form(key="chat_form", clear_on_submit=True):
    user_question = st.text_input("Ask a question about your graph:", "")
    submitted = st.form_submit_button("Send")


# Helper to keep only last 5 Q&A
def trim_history():
    if len(st.session_state.chat_history) > 5:
        st.session_state.chat_history = st.session_state.chat_history[-5:]


# Run Graph RAG and update chat history
if submitted and user_question.strip():
    # Prepare additional_context: last 5 Q&A
    additional_context = [
        {"question": entry["question"], "answer": entry["answer"]}
        for entry in st.session_state.chat_history[-5:]
    ]
    # Run Graph RAG (async)
    with st.spinner("Generating answer..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            run_graphrag({"question": user_question, "additional_context": additional_context})
        )
        loop.close()
    st.session_state.chat_history.append(result)
    trim_history()

# Display chat history (most recent at top)
for i, entry in enumerate(reversed(st.session_state.chat_history)):
    with st.container():
        st.markdown(f"**Q{len(st.session_state.chat_history)-i}:** {entry['question']}")
        st.markdown(f"**A{len(st.session_state.chat_history)-i}:** {entry['answer']}")
        st.markdown("**Generated Cypher:**")
        st.code(entry["cypher"], language="cypher")
