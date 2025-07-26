from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import asyncio

load_dotenv()

text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_name="csv-ai"

async def load_split_store_document(file_path:str,file_id:int):
    if file_path.endswith('.csv'):
        loader = CSVLoader(file_path, encoding="utf-8")
    elif file_path.endswith('.csv'):
        loader = CSVLoader(file_path, encoding="cp1252")
    elif file_path.endswith('.xls'):
        loader = UnstructuredExcelLoader(file_path)
    elif file_path.endswith('.xlsx'):
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = loader.load()
    for doc in documents:
        doc.metadata["file_id"]=str(file_id)
        
    docs = text_spliter.split_documents(documents)
    
    def _store():
        try:
            vectorstore=PineconeVectorStore(
                index_name=index_name,
                embedding=embedding,
                namespace="default"
            )
            vectorstore.add_documents(docs)
            print("✅ Documents added to Pinecone.")
            return True
        except Exception as e:
            print("❌ Error in _store thread:", str(e))
            return False

    result = await asyncio.to_thread(_store)
    return result

    
    # vector_store_docs.add_documents(docs)
    # retriever=vector_store_docs.as_retriever(search_type="similarity", search_kwargs={"k":6})
    # return retriever

async def delete_doc_from_pinecone(file_id:int):
    def _delete():
        try:
            vectorstore=PineconeVectorStore(
                index_name=index_name,
                embedding=embedding
            )
            vectorstore.delete(filter={"file_id":str(file_id)})
            print(f"Deleted all documents with file_id={file_id}")
            return True
        except Exception as e:
            print(f"Error deleting document with file_id: {file_id} from chroma: {str(e)}")
            return False
    return await asyncio.to_thread(_delete)

# asyncio.run(load_split_store_document(file_path="carprices.csv", file_id="carprices.csv"))

load_split_store_document(file_path="carprices.csv", file_id="carprices.csv")
