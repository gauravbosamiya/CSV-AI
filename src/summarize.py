import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader

load_dotenv()

def summarize(model_name, api_key):
    st.write("# Summary of CSV/Excel")
    st.write("Upload your document here.")

    uploaded_file = st.file_uploader("Upload source document", type=["csv", "xls", "xlsx"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            st.info("📥 Saving uploaded file...")

            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(uploaded_file.getvalue())
                temp_file_path = f.name
            
            st.info("📄 Reading file content...")

            try:
                loader = CSVLoader(file_path=temp_file_path, encoding="utf-8")
                data = loader.load()
            except:
                loader = UnstructuredExcelLoader(file_path=temp_file_path)
                data = loader.load()

            os.remove(temp_file_path)
            
            st.info("🔍 Splitting content into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(data)

            st.success("✅ File ready for summarization!")

        except Exception as e:
            st.error(f"❌ Error during upload or parsing: {e}")
            return

        if st.button("✨ Generate Summary"):
            try:
                with st.spinner("Generating summary..."):
                    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
                    chain = load_summarize_chain(
                        llm=llm,
                        chain_type="map_reduce",
                        return_intermediate_steps=True,
                        input_key="input_documents",
                        output_key="output_text",
                    )
                    result = chain({"input_documents": texts}, return_only_outputs=True)
                    
                    st.success(result['output_text'])
            except Exception as e:
                st.warning("❌ Failed to generate summary. Please check your API key and model.")
                st.stop()
