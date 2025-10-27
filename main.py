from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uvicorn

from rag_utils import load_vector_store, get_answer

 
from fastapi import UploadFile, File

app = FastAPI()

# Initialiser le retriever
vector_store, retriever = load_vector_store()

# Schéma de question texte
class Question(BaseModel):
    query: str

# Endpoint pour chatbot texte
@app.post("/ask")
async def ask_question(q: Question):
    response = get_answer(q.query, retriever)
    return JSONResponse(content={"answer": response}, media_type="application/json; charset=utf-8")




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
