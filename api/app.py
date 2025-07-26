from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
from api.api_utils import load_split_store_document, delete_doc_from_pinecone
import shutil
import uuid
import asyncio

app = FastAPI()


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
        
    