import os
import requests
import numpy as np
from fastapi import FastAPI, Query
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---- EMBEDDING HELPER ----
def embed_texts(texts):
    """Generate embeddings using OpenAI embedding model."""
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in res.data]


# ---- COSINE SIMILARITY ----
def cos_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---- FETCH MESSAGES ----
def fetch_messages(total_limit=400):
    """Fetch messages from the API using pagination."""
    url = "https://november7-730026606190.europe-west1.run.app/messages"
    all_msgs = []
    skip = 0
    limit = 100

    while len(all_msgs) < total_limit:
        resp = requests.get(url, params={"skip": skip, "limit": limit})
        if resp.status_code != 200:
            break

        data = resp.json()
        items = data.get("items", [])
        if not items:
            break

        all_msgs.extend(items)
        skip += limit

    return all_msgs[:total_limit]


# ---- MAIN ENDPOINT ----
@app.get("/ask")
def ask(question: str = Query(...)):
    # Fetch database messages
    messages = fetch_messages()

    if not messages:
        return {"answer": "Could not fetch messages."}

    docs = [msg["message"] for msg in messages]
    users = [msg["user_name"] for msg in messages]

    # Embed question + docs using OpenAI
    q_emb = embed_texts([question])[0]
    d_embs = embed_texts(docs)

    # Compute similarity for each message
    scores = [cos_sim(q_emb, d) for d in d_embs]
    best_idx = int(np.argmax(scores))

    top_matches = sorted(
        list(zip(scores, docs, users)),
        key=lambda x: x[0],
        reverse=True
    )[:5]

    # Build a retrieval summary for GPT
    context = "\n".join([f"- {u}: {m}" for _, m, u in top_matches])

    # Ask GPT to answer using only retrieved messages
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer using ONLY the information in the retrieved messages."},
                {"role": "user", "content": f"Question: {question}\n\nRetrieved messages:\n{context}"}
            ]
        )

        final_answer = completion.choices[0].message.content

        return {
            "answer": final_answer,
            "retrieved_messages": [
                {"user": u, "message": m} for _, m, u in top_matches
            ]
        }

    except Exception as e:
        return {
            "answer": "OpenAI error occurred",
            "error": str(e),
            "retrieved_messages": [
                {"user": u, "message": m} for _, m, u in top_matches
            ]
        }








