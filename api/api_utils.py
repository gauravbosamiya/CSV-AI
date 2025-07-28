import asyncio
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Constants
INDEX_NAME = "csv-ai"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True
)

@st.cache_resource()
async def load_split_store_document(uploaded_file, file_id: int):
    """Load, split, and store document in Pinecone with file_id metadata."""
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_file_path = tmp.name

    # Load file based on extension
    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                loader = CSVLoader(file_path=temp_file_path, encoding="utf-8")
                documents = loader.load()
            except Exception:
                loader = CSVLoader(file_path=temp_file_path, encoding="cp1252")
                documents = loader.load()
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            loader = UnstructuredExcelLoader(file_path=temp_file_path)
            documents = loader.load()
        else:
            raise ValueError(f"❌ Unsupported file type: {uploaded_file.name}")
    except Exception as e:
        st.error(f"❌ Failed to load file: {str(e)}")
        raise e

    # Split and add metadata
    docs = text_splitter.split_documents(documents)
    for doc in docs:
        doc.metadata["file_id"] = str(file_id)

    # Store in Pinecone
    def _store():
        try:
            vectorstore = PineconeVectorStore.from_documents(
                documents=docs,
                embedding=embedding,
                index_name=INDEX_NAME,
            )
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            print("✅ Documents added to Pinecone.")
            return vectorstore, retriever
        except Exception as e:
            print("❌ Error storing documents:", str(e))
            raise e

    return await asyncio.to_thread(_store)

@st.cache_resource()
async def delete_doc_from_pinecone(file_id: int) -> bool:
    """Delete all documents from Pinecone with a specific file_id."""
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def _delete():
        try:
            vectorstore = PineconeVectorStore(
                index_name=INDEX_NAME,
                embedding=embedding
            )
            vectorstore.delete(filter={"file_id": str(file_id)})

            print(f"✅ Successfully deleted documents with file_id = {file_id}")
            return True
        except Exception as e:
            print(f"❌ Error deleting file_id = {file_id} from Pinecone: {e}")
            return False

    return await asyncio.to_thread(_delete)
# asyncio.run(load_split_store_document(file_path="carprices.csv", file_id="carprices.csv"))

# load_split_store_document(file_path="carprices.csv", file_id="carprices.csv")



# format_hint = """
# Please format your answer using **Markdown**:
# - Use bullet points for lists
# - Use tables for tabular data
# - Wrap code (e.g. Python, SQL) in triple backticks (```python)
# - Use headings (###) for section titles
# """

# chat_prompt=ChatPromptTemplate.from_messages([
#     ("system","You are an intelligent AI assistant who can both remember context and analyze data. "
#                "If user ask the question which is not related to csv,xls,xlsx file just say i don't have access."),
    
# ])