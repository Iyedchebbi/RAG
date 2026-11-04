from pathlib import Path
import requests
import numpy as np

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"


# Generate embeddings
def embed(texts):
    response = requests.post(
        f"{OLLAMA}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        }
    )
    response.raise_for_status()
    return np.array(response.json()["embeddings"])


# Ask the LLM
def ask_llm(prompt):
    response = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

chunks=[]
for file in Path('docs').glob('*.md'):
    chunks +=[p.strip() for p in file.read_text(encoding="utf-8").split("\n\n") if p.strip()]

chunk_vectors = embed (chunks)
print(f"Indexed {len(chunks)} chunks\n")


def retrieve(question , k=3):
    q=embed([question])[0]
    scores = chunk_vectors @ q/ (np.linalg.norm(chunk_vectors,axis = 1 )* np.linalg.norm(q))
    return [chunks[i] for i in np.argsort(scores)[::-1][:k]]


while True:
    question = input("Ask : ")
    context = "\n\n".join(retrieve(question))
    print(context)
    prompt = f"""answer the question using only the context below.
    if the answer isn't in the context, say "i don't know".
Context:
{context}
Question: {question} """

    print("\n" + ask_llm(prompt)+ "\n")
