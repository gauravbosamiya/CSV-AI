from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
from api.api_utils import load_split_store_document, delete_doc_from_pinecone
import shutil
import uuid
import asyncio
import tempfile
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain


app = FastAPI()

@app.get("/")
def home():
    return {'message':'Welcome to the CSV-AI API !!!'}


@app.get("/about")
def about():
    return {'message':'CSV-AI is a tool where you can Interacting, Analyzing and Summarizing CSV Files'}

@app.post('/upload-csv')
async def upload_csv(file:UploadFile=File(...)):
    allowed_type = ['.csv','.xls','.xlsx']
    
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_type:
        raise HTTPException(status_code=400, detail=f"Unsupported file type Allowed file types are : {', '.join(allowed_type)}")

    temp_file_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
    with open(temp_file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        file_id=file.filename
        success=await load_split_store_document(temp_file_path, file_id)
        if success:
            return JSONResponse(content={"message": "File processed and stored in database successfully."})
        else:
            await delete_doc_from_pinecone(file_id)
            raise HTTPException(status_code=500, detail=f"failed to index {file.filename}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
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
        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )
        result = chain({"input_documents": texts}, return_only_outputs=True)
        return JSONResponse(content={"summary": result["output_text"]})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    

