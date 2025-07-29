import shutil
import tempfile
import asyncio
from typing import Tuple
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    UnstructuredExcelLoader,
    PyMuPDFLoader,
    Docx2txtLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "csv-ai"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True
)

async def load_split_store_document(file_obj, file_id: str) -> Tuple[PineconeVectorStore, object]:
    filename = getattr(file_obj, "filename", None)
    if not filename:
        raise ValueError("❌ Uploaded file has no valid filename.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp:
        shutil.copyfileobj(file_obj.file, tmp)
        temp_file_path = tmp.name

    loader = None
    if filename.endswith(".csv"):
        try:
            loader = CSVLoader(file_path=temp_file_path, encoding="utf-8")
            documents = loader.load()
        except Exception:
            loader = CSVLoader(file_path=temp_file_path, encoding="cp1252")
            documents = loader.load()
    elif filename.endswith((".xls", ".xlsx")):
        loader = UnstructuredExcelLoader(file_path=temp_file_path)
        documents = loader.load()
    elif filename.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path=temp_file_path)
        documents = loader.load()
    elif filename.endswith(".docx"):
        loader = Docx2txtLoader(file_path=temp_file_path)
        documents = loader.load()
    else:
        raise ValueError(f"❌ Unsupported file type: {filename}")

    chunks = text_splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["file_id"] = file_id

    def _store():
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embedding,
            index_name=INDEX_NAME,
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        return vectorstore, retriever

    return await asyncio.to_thread(_store)


async def delete_doc_from_pinecone(file_id: str) -> bool:
    def _delete():
        try:
            vectorstore = PineconeVectorStore(
                index_name=INDEX_NAME, 
                embedding=embedding
            )
            vectorstore.delete(filter={"file_id": file_id})
            return True
        except Exception:
            return False

    return await asyncio.to_thread(_delete)


async def load_retriever_by_file_id(file_id: str):
    def _load():
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME, 
            embedding=embedding
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": {"file_id": file_id}, "k": 5}
        )
        return retriever

    return await asyncio.to_thread(_load)


