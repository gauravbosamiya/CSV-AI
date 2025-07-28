from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
from api.api_utils import load_split_store_document, delete_doc_from_pinecone, load_retriever_by_file_id
import uuid
import tempfile
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
INDEX_NAME = "csv-ai"



load_dotenv()

app = FastAPI()

@app.get("/")
def home():
    return {'message':'Welcome to the CSV-AI API !!!'}


@app.get("/about")
def about():
    return {'message':'CSV-AI is a tool where you can Interacting, Analyzing and Summarizing CSV Files'}


@app.post("/summarize")        
async def summarize(file:UploadFile=File(...), model_name:str=Form(...),api_key:str=Form(...)):
    try:
        contents=await file.read()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(contents)
            temp_file_path=f.name
        
        try:
            loader = CSVLoader(temp_file_path)
            data =loader.load()
        except:
            loader = UnstructuredExcelLoader(temp_file_path)
            data=loader.load()
            
            
        os.remove(temp_file_path)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        texts=text_splitter.split_documents(data)
        
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        
        
        structured_summary_prompt = PromptTemplate.from_template("""
        You are a helpful AI assistant. Summarize the following content in a **structured and detailed Markdown format**.

        The summary should include:
        - A high-level overview of the dataset
        - Key statistics (means, counts, etc.)
        - Notable trends or patterns
        - Anomalies or issues
        - Final conclusion

        If possible, use:
        - `###` headings for sections
        - Bullet points
        - Code blocks (```python) for examples
        - Tables if relevant

        --- Begin Dataset Content ---
        {input}
        --- End Dataset Content ---

        Return only the structured summary.
        """)
        # chain = structured_summary_prompt | llm
        
        chain = LLMChain(llm=llm, prompt=structured_summary_prompt)
        input_text = "\n".join([t.page_content for t in texts])
        result = chain.run(input=input_text)
        return JSONResponse(content={"summary": result})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}") 
    

@app.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...)):
    if not file.filename.endswith((".csv", ".xls", ".xlsx", ".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload CSV / Excel / PDF / DOCX")

    file_id = str(uuid.uuid4())

    try:
        await load_split_store_document(file, file_id)
        return JSONResponse({"message": "✅ File processed successfully.", "file_id": file_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.delete("/delete-doc/{file_id}")
async def delete_doc(file_id: str):
    try:
        success = await delete_doc_from_pinecone(file_id)
        if success:
            return JSONResponse({"message": f"✅ Document with file_id={file_id} deleted."})
        else:
            raise HTTPException(status_code=400, detail="Document not found or failed to delete.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    
    
class QueryInput(BaseModel):
    query: str

@app.post("/load-retriever/{file_id}")
async def retrieve_chunks(file_id: str, body: QueryInput):
    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedding)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "filter": {"file_id": file_id}}
        )

        docs = retriever.invoke(body.query)

        if not docs:
            return {"context": ["No relevant information found in the document for your query."]}

        chunks = [doc.page_content for doc in docs]
        return {"context": chunks}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
