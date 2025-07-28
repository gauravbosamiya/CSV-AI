import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import requests


load_dotenv()

def summarize(model_name, api_key, api_url="http://localhost:8000/summarize/"):
    st.write("# Summary of CSV/Excel")
    st.write("Upload your document here.")

    uploaded_file = st.file_uploader("Upload source document", type=["csv", "xls", "xlsx"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.info("📥 File uploaded successfully.")
        
        if st.button("✨ Generate Summary"):
            try:
                st.info("🔄 Sending file to API for summarization...")

                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp:
                    temp.write(uploaded_file.getvalue())
                    temp_path = temp.name

                with open(temp_path, "rb") as f:
                    files = {"file": (uploaded_file.name, f, uploaded_file.type)}
                    data = {
                        "model_name": model_name,
                        "api_key": api_key
                    }
                    response = requests.post(api_url, files=files, data=data)

                os.remove(temp_path)

                if response.status_code == 200:
                    summary = response.json().get("summary")
                    st.success(summary)
                else:
                    st.error(f"❌ API Error: {response.json().get('error')}")

            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
                
                

