from fastapi import FastAPI
from pydantic import BaseModel
from rag import RAGEngine

app = FastAPI()
# Initialize the engine (Make sure the CSV path is correct)
engine = RAGEngine('cleaned_data.csv')

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_law(query: Query):
    answer = engine.get_answer(query.question)
    return {"answer": answer}

