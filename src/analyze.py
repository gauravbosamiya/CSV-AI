import streamlit as st
from langchain_experimental.agents import create_pandas_dataframe_agent
import tempfile
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory 


load_dotenv()

def analyze(model_name,api_key):
    st.write("# Analyze CSV & Excel")
    reset = st.sidebar.button("Reset Chat")
    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV or Excel file:", type=["csv", "xls", "xlsx"])
    if not uploaded_file:
        st.warning("Please upload a file to continue.")
        return
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    try:
        if uploaded_file.name.endswith('.csv'):
            df=pd.read_csv(tmp_file_path)
        else:
            df=pd.read_excel(tmp_file_path)
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        return
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    if "memory" not in st.session_state:
        st.session_state.memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
    agent_executor=create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        memory=st.session_state.memory,
        verbose=True,
    )
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything about your data."}]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if prompt := st.chat_input("Ask me anything about your dataset..."):
        if not api_key:
            st.warning("Please provide your Google API key.")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        try:
            response = agent_executor.invoke({"input": prompt})
            output = response["output"]
            st.session_state.messages.append({"role": "assistant", "content": output})

            st.chat_message("assistant").write(output)
        except Exception as e:
            st.error(f"❌ Error: {e}")
    if reset:
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.rerun()

