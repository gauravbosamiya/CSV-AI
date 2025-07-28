import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import uuid
import asyncio
from api.api_utils import load_split_store_document, delete_doc_from_pinecone
from api.db import get_history, delete_history, save_history
from dotenv import load_dotenv

load_dotenv()

def chat(model_name, api_key):
    st.write("# Talk with CSV/Excel")

    # Initialize Session State
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        history = get_history(st.session_state.session_id)
        st.session_state.messages = history or [{"role": "assistant", "content": "How can I help you?"}]

    # Reset Chat
    if st.sidebar.button("🔄 Reset Chat"):
        delete_history(st.session_state.session_id)
        st.session_state.messages = [{"role": "assistant", "content": "Chat reset. How can I help you?"}]

    # File Upload
    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV or Excel", type=["csv", "xls", "xlsx"])

    # Assign a unique file_id once
    if uploaded_file and "file_id" not in st.session_state:
        st.session_state.file_id = str(uuid.uuid4())
        st.session_state.doc_deleted = False

    file_id = st.session_state.get("file_id")

    # Delete document from Pinecone
    if st.sidebar.button("❌ Delete Uploaded Document"):
        if file_id:
            with st.spinner("Deleting document from Pinecone..."):
                success = asyncio.run(delete_doc_from_pinecone(file_id))
            if success:
                st.success(f"✅ Document with file_id={file_id} deleted.")
                st.session_state.pop("retriever", None)
                st.session_state.pop("file_id", None)
                st.session_state["doc_deleted"] = True
            else:
                st.error("❌ Failed to delete document.")
        else:
            st.warning("⚠️ No document to delete.")

    # Load and process document
    if uploaded_file and "retriever" not in st.session_state and not st.session_state.get("doc_deleted", False):
        with st.spinner("🔍 Processing file..."):
            try:
                vectorstore, retriever = asyncio.run(load_split_store_document(uploaded_file, file_id))
                st.session_state.retriever = retriever
                st.success("✅ File processed. Ask your questions!")
            except Exception as e:
                st.error(f"❌ File processing failed: {str(e)}")
                return

    retriever = st.session_state.get("retriever")

    # Display previous messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask your question about the dataset"):
        if not retriever:
            st.warning("⚠️ Please upload and process a file first.")
            return

        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            docs = retriever.invoke(prompt)
            context = "\n\n".join([d.page_content for d in docs])
        except Exception as e:
            st.error(f"❌ Retrieval failed: {str(e)}")
            return

        # LLM Prompt Formatting
        format_hint = """
        Please format your answer using **Markdown**:
        - Use bullet points for lists
        - Use tables for tabular data
        - Wrap code (e.g. Python, SQL) in triple backticks (```python)
        - Use headings (###) for section titles
        """

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful data assistant. {format_hint}"),
            ("user", "Context:\n{context}\n\nQuestion: {question}")
        ])

        llm = GoogleGenerativeAI(model=model_name, google_api_key=api_key)
        chain = prompt_template | llm

        try:
            response = chain.invoke({"context": context, "question": prompt})
            answer = response.content if hasattr(response, "content") else str(response)
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            save_history(st.session_state.session_id, st.session_state.messages)
        except Exception as e:
            st.error(f"❌ LLM error: {str(e)}")
