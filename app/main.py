import streamlit as st 
from dotenv import load_dotenv
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --------------------------------------------------------------
# gemini-2.5-flash
# gemini-2.5-pro
# ---------------------------------------------------------------
# model  = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
# result = model.invoke("Hiii")
# print(result.content)
# --------------------------------------------------------------

st.set_page_config(page_title="CSV AI", page_icon="🧾", layout="wide")


st.sidebar.subheader("Enter your Groq API key...")
st.sidebar.text_input(label="",placeholder="GROQ API key...",type="password")

# def main():
#     st.markdown(
#         """
#         <div style='text-align: center;'>
#             <h1>🧠 CSV AI</h1>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
#     st.markdown(
#         """
#         <div style='text-align: center;'>
#             <h4>⚡️ Interacting, Analyzing and Summarizing CSV Files!</h4>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )