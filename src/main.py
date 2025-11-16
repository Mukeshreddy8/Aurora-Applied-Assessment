from fastapi import FastAPI
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load local .env (ignored in Render but useful locally)
load_dotenv()

app = FastAPI()

# Create OpenAI client with explicit API key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
print("DEBUG OPENAI KEY PREFIX:", (OPENAI_KEY[:7] if OPENAI_KEY else "None"))

client = OpenAI(api_key=OPENAI_KEY)


# ------------------------------
# Fetch messages (with pagination)
# ------------------------------
def fetch_messages(limit=100):
    url = "https://november7-730026606190.europe-west1.run.app/messages"

    resp = requests.get(url, params={"skip": 0, "limit": limit})

    print("STATUS:", resp.status_code)
    print("RAW TEXT:", resp.text[:300])

    if resp.status_code != 200:
        return None

    try:
        data = resp.json()
        return data.get("items", [])
    except:
        return None


# ------------------------------
# Embed text using OpenAI
# ------------------------------
def embed_text(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return res.data[0].embedding


def embed_texts(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in res.data]


# ------------------------------
# Simple cosine similarity
# ------------------------------
def cosine_similarity(a, b):
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ------------------------------
# /ask endpoint
# ------------------------------
@app.get("/ask")
def ask(question: str):
    # Fetch messages
    messages = fetch_messages(limit=300)
    if not messages:
        return {"answer": "Could not fetch messages."}

    # Prepare message list
    contents = [m["message"] for m in messages]
    users = [m["user_name"] for m in messages]

    # Embed query + messages
    q_emb = embed_text(question)
    m_embs = embed_texts(contents)

    # Compute similarity
    scores = [cosine_similarity(q_emb, emb) for emb in m_embs]
    best_idx = scores.index(max(scores))

    # Build LM answer with context
    best_message = contents[best_idx]
    best_user = users[best_idx]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer concisely based ONLY on the provided message."
                },
                {
                    "role": "user",
                    "content": f"QUESTION: {question}\n\nCONTEXT: {best_message}"
                },
            ]
        )

        answer = completion.choices[0].message.content

        return {
            "answer": answer,
            "retrieved_message": best_message,
            "user": best_user
        }

    except Exception as e:
        return {
            "answer": "OpenAI error occurred",
            "error": str(e),
            "retrieved_message": best_message,
            "user": best_user
        }









