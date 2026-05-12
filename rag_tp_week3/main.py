from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import RAGEngine # Assuming you use the RAGEngine class from previous step

app = FastAPI()

# CRITICAL: This allows your HTML file to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine('researsh/cleaned_data.csv')

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_law(query: Query):
    answer = engine.get_answer(query.question)
    return {"answer": answer}