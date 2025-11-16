from fastapi import FastAPI, Query
import requests
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

MESSAGES_API = "https://november7-730026606190.europe-west1.run.app/messages"


def fetch_all_messages(limit=2000):
    """Fetch dataset in batches using skip/limit pagination."""
    messages = []
    skip = 0

    while True:
        resp = requests.get(MESSAGES_API, params={"skip": skip, "limit": 200})
        data = resp.json()

        batch = data.get("items", [])
        if not batch:
            break

        messages.extend(batch)
        skip += 200

        if skip >= limit:
            break

    return messages


@app.get("/ask")
def ask(question: str = Query(...)):
    # Step 1: Fetch messages
    messages = fetch_all_messages()

    docs = [msg["message"] for msg in messages]
    names = [msg["user_name"] for msg in messages]

    # Step 2: Semantic retrieval (top 5)
    q_emb = model.encode(question, convert_to_tensor=True)
    d_embs = model.encode(docs, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(q_emb, d_embs)[0]
    top_k = scores.topk(5)

    retrieved = []
    for idx in top_k.indices:
        idx = int(idx)
        retrieved.append({
            "user": names[idx],
            "message": docs[idx]
        })

    # Step 3: LLM summarization
    prompt = f"""
You are an assistant answering a question using ONLY the retrieved messages.
If the information is not present, reply: "I could not find that information."

Question: {question}

Messages:
{retrieved}

Provide a short, factual answer.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )

        # FIXED: correct way to access message
        final_answer = completion.choices[0].message.content

    except Exception as e:
        return {
            "answer": "OpenAI error occurred",
            "error": str(e),
            "retrieved_messages": retrieved
        }

    return {"answer": final_answer}








