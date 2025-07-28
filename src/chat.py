import streamlit as st
import uuid
import requests
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from api.db import get_history, delete_history, save_history

load_dotenv()

UPLOAD_URL = "http://localhost:8000/upload-doc"
DELETE_URL = "http://localhost:8000/delete-doc"
RETRIEVER_URL = "http://localhost:8000/load-retriever"


def chat(model_name, api_key):
    st.title("📊 Talk with CSV / Excel / PDF / DOCX Files")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "file_id" not in st.session_state:
        st.session_state.file_id = None

    if "retriever_ready" not in st.session_state:
        st.session_state.retriever_ready = False

    if "messages" not in st.session_state:
        history = get_history(st.session_state.session_id)
        if history and isinstance(history, list):
            st.session_state.messages = history
        else:
            st.session_state.messages = [{"role": "assistant", "content": "Hi! Upload a file and ask me anything about it."}]

    session = st.session_state

    with st.sidebar:
        st.header("📂 File & Session Controls")

        if st.button("🔄 Reset Chat"):
            delete_history(session.session_id)
            session.messages = [{"role": "assistant", "content": "✅ Chat reset. Upload a new file to begin."}]
            session.file_id = None
            session.retriever_ready = False

        uploaded_file = st.file_uploader("📁 Upload CSV, Excel, PDF, or DOCX", type=["csv", "xls", "xlsx", "pdf", "docx"])
        if uploaded_file and not session.file_id:
            session.file_id = str(uuid.uuid4())
            with st.spinner("Uploading and processing file..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(UPLOAD_URL, files=files)
                    if response.status_code == 200:
                        result = response.json()
                        session.file_id = result["file_id"]
                        session.retriever_ready = True
                        st.success("✅ Document processed and ready!")
                    else:
                        st.error("❌ Upload failed: " + response.text)
                        session.file_id = None
                except Exception as e:
                    st.error(f"❌ Upload error: {str(e)}")
                    session.file_id = None

        if st.button("❌ Delete Uploaded Document"):
            if session.file_id:
                with st.spinner("Deleting from vector DB..."):
                    try:
                        response = requests.delete(f"{DELETE_URL}/{session.file_id}")
                        if response.status_code == 200:
                            st.success("✅ Document deleted.")
                            session.file_id = None
                            session.retriever_ready = False
                        else:
                            st.error("❌ Deletion failed: " + response.text)
                    except Exception as e:
                        st.error(f"❌ Error deleting file: {str(e)}")
            else:
                st.warning("⚠️ No uploaded file to delete.")

        with st.expander("🪪 Debug Info"):
            st.markdown(f"**Session ID**: `{session.session_id}`")
            st.markdown(f"**File ID**: `{session.file_id}`")
            st.markdown(f"**Messages**: `{len(session.messages)}`")

    for msg in session.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("Ask your question here..."):
        if not session.retriever_ready:
            st.warning("⚠️ Please upload and process a file first.")
            return

        st.chat_message("user").markdown(prompt)
        session.messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(f"{RETRIEVER_URL}/{session.file_id}", json={"query": prompt})
            response.raise_for_status()
            context_chunks = response.json().get("context", [])
            context = "\n\n".join(context_chunks)
        except Exception as e:
            st.error(f"❌ Retrieval error: {str(e)}")
            return

        history_str = ""
        for m in session.messages:
            if m["role"] in ["user", "assistant"]:
                prefix = "User" if m["role"] == "user" else "Assistant"
                history_str += f"{prefix}: {m['content']}\n"

        format_hint = """
Please format your answer using **Markdown**:
- Use bullet points for lists
- Use tables for tabular data
- Wrap code (e.g. Python, SQL) in triple backticks (```python)
"""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful data assistant. Use the following conversation history and file context to answer the user's question. {format_hint}"),
            ("user", "History:\n{history}\n\nContext:\n{context}\n\nQuestion: {question}")
        ])

        try:
            llm = GoogleGenerativeAI(model=model_name, google_api_key=api_key)
            chain = prompt_template | llm
            result = chain.invoke({
                "history": history_str,
                "context": context,
                "question": prompt
            })

            answer = getattr(result, "content", str(result))
            st.chat_message("assistant").markdown(answer)
            session.messages.append({"role": "assistant", "content": answer})

            save_history(session.session_id, session.messages)

        except Exception as e:
            st.error(f"❌ LLM Error: {str(e)}")
