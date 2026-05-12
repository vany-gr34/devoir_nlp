import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class RAGEngine:
    def __init__(self, data_path, model_name="microsoft/Phi-4-mini-instruct"):
        # Load Data
        self.df = pd.read_csv(data_path)
        self.texts = self.df["infraction_desc"].tolist()
        
        # Load Embedding Model
        self.embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Build FAISS Index
        embeddings = self.embed_model.encode(self.texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))
        
        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.generator = pipeline("text-generation", model=self.llm, tokenizer=self.tokenizer)

    def retrieve(self, query, k=3):
        query_vec = self.embed_model.encode([query])
        distances, indices = self.index.search(np.array(query_vec), k)
        return [self.texts[i] for i in indices[0]]

    def get_answer(self, query):
        docs = self.retrieve(query)
        context = "\n".join(docs)
        prompt = f"""Role: Moroccan traffic law expert.
Instructions:
- Answer strictly from the context.
- Do not add external knowledge.
- If missing info, say: "لا أملك معلومات كافية في السياق."
- Answer in Arabic only.
- Keep answers short and precise.

Context:
{context}

User Question:
{query}

Final Answer:
"""
        output = self.generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.3)
        # Extract only the Final Answer part
        full_text = output[0]["generated_text"]
        return full_text.split("Final Answer:")[-1].strip()