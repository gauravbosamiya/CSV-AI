import streamlit as st
import pandas as pd
import tempfile
import uuid
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

def analyze(model_name, api_key):
    st.set_page_config(page_title="CSV AI Agent", layout="wide")
    st.title("🧠 CSV & Excel Agent")

    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xls", "xlsx"])
    reset = st.sidebar.button("Reset Chat")

    if not uploaded_file:
        st.warning("Please upload a file to continue.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(tmp_file_path)
        else:
            df = pd.read_excel(tmp_file_path)
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        return

    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    base_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        allow_dangerous_code=True,
        verbose=False,
    )

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    history = StreamlitChatMessageHistory(key="chat_messages")

    for msg in history.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.chat_message(role).write(msg.content)

    if prompt := st.chat_input("Ask something about your data or chat normally..."):
        st.chat_message("user").write(prompt)
        history.add_user_message(prompt)


        format_hint = """
        Please format your answer using **Markdown**:
        - Use bullet points for lists
        - Use tables for tabular data
        - Wrap code (e.g. Python, SQL) in triple backticks (```python)
        - Use headings (###) for section titles
        """

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an intelligent AI assistant who can both remember context and analyze data. If user ask the question which is not related to csv,xls,xlsx file just say i don't have access you can only ask about your csv,xls,xlsx."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}\n\n" + format_hint)
        ])
        previous_turns = [
            HumanMessage(content=m.content) if isinstance(m, HumanMessage) else AIMessage(content=m.content)
            for m in history.messages[:-1] 
        ]
        chat_messages = chat_prompt.format_messages(
            history=previous_turns,
            user_input=prompt
        )
        with st.spinner("Generating answer..."):
            try:
                response = base_agent.invoke(chat_messages)
                history.add_ai_message(response['output'])
                st.chat_message("assistant").write(response['output'])

            except Exception as e:
                st.error(f"❌ Error: {e}")

    if reset:
        history.clear()
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
