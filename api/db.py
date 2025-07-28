from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI=os.getenv("MONGO_URI")
client=MongoClient(MONGO_URI)
db=client["CSV-AI"]
collection=db["users_history"]

def get_history(session_id:str):
    doc=collection.find_one({"session_id":session_id})
    return doc["history"] if doc else []

def save_history(session_id:str, history:list):
    collection.update_one(
        {"session_id":session_id},
        {"$set":{"history":history}},
        upsert=True
    )
def delete_history(session_id:str):
    collection.delete_one({"session_id":session_id})