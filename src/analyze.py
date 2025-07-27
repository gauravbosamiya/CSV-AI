# import streamlit as st
# from langchain_experimental.agents import create_pandas_dataframe_agent
# import tempfile
# import pandas as pd
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# from langchain.memory import ConversationBufferMemory 


# load_dotenv()

# def analyze(model_name,api_key):
#     st.write("### Analyze CSV & Excel🧾")
    
#     uploaded_file = st.sidebar.file_uploader("📁 Upload CSV or Excel file:", type=["csv", "xls", "xlsx"])
#     reset = st.sidebar.button("Reset Chat")
#     if not uploaded_file:
#         st.warning("Please upload a file to continue.")
#         return
    
#     with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
#         tmp_file.write(uploaded_file.getvalue())
#         tmp_file_path = tmp_file.name
        
#     try:
#         if uploaded_file.name.endswith('.csv'):
#             df=pd.read_csv(tmp_file_path)
#         else:
#             df=pd.read_excel(tmp_file_path)
#     except Exception as e:
#         st.error(f"❌ Failed to read file: {e}")
#         return
    
#     llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    
#     if "memory" not in st.session_state:
#         st.session_state.memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
#     if "messages" not in st.session_state:
#         st.session_state.messages = [{
#             "role": "assistant", "content": "Hi! Ask me anything about your data."
#         }]


#     agent_executor=create_pandas_dataframe_agent(
#         llm=llm,
#         df=df,
#         memory=st.session_state.memory,
#         verbose=False,
#         allow_dangerous_code=True,
#     )
    
#     # if "messages" not in st.session_state:
#     #     st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything about your data."}]
    
#     for msg in st.session_state.messages:
#         st.chat_message(msg["role"]).write(msg["content"])
        
#     if prompt := st.chat_input("Ask me anything about your dataset..."):
#         if not api_key:
#             st.warning("Please provide your Google API key.")
#             st.stop()
            
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         st.chat_message("user").write(prompt)

#         try:
#             response = agent_executor.invoke({"input": prompt})
#             output = response["output"]
            
#             st.session_state.messages.append({"role": "assistant", "content": output})
#             st.chat_message("assistant").write(output)
            
#         except Exception as e:
#             st.error(f"❌ Error: {e}")
            
#     if reset:
#         st.session_state.messages = []
#         st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         st.rerun()

# ---------------memory------------------
import streamlit as st
import pandas as pd
import tempfile
import uuid
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

def analyze(model_name, api_key):
    st.set_page_config(page_title="CSV AI Agent", layout="wide")
    st.title("🧠 CSV & Excel Agent")

    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xls", "xlsx"])
    reset = st.sidebar.button("Reset Chat")

    if not uploaded_file:
        st.warning("Please upload a file to continue.")
        return

    # Read file to DataFrame
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

    # Create LLM
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    # Create Pandas DataFrame Agent (no memory)
    base_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        allow_dangerous_code=True,
        verbose=False,
    )

    # Session ID for user
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Chat message history using BaseMessage
    history = StreamlitChatMessageHistory(key="chat_messages")

    # Display chat history
    for msg in history.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.chat_message(role).write(msg.content)

    # Prompt input
    if prompt := st.chat_input("Ask something about your data or chat normally..."):
        st.chat_message("user").write(prompt)
        history.add_user_message(prompt)

        # Compose a history-aware prompt (acts like memory)
        previous_turns = "\n".join([f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}" for m in history.messages[:-1]])

        full_prompt = f"""
            You are an intelligent AI assistant who can both remember context and analyze data.
            Below is the recent conversation:
            {previous_turns}

            Now the user says: {prompt}

            Respond appropriately using context and data knowledge.
            """

        try:
            response = llm.invoke(full_prompt)
            history.add_ai_message(response.content)
            st.chat_message("assistant").write(response.content)

        except Exception as e:
            st.error(f"❌ Error: {e}")

    # Reset button
    if reset:
        history.clear()
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
