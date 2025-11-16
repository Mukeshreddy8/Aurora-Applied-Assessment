README — Member Question Answering API
📌 Overview
This project implements a question-answering API that can answer natural-language questions about member messages using:

Semantic Retrieval (Sentence Transformers)
LLM Summarization (OpenAI GPT-4o-mini)
FastAPI as the backend framework
The system retrieves the most relevant messages based on the question and uses an LLM to summarize or extract the final answer strictly from the retrieved messages.
The API follows the exact assignment requirement:

Respond using ONLY the content present in the dataset.
If the answer is not available, return: “I could not find that information.”


🚀 Features
✔ Semantic search over all member messages
✔ Top-5 message retrieval using cosine similarity
✔ OpenAI-powered summarization (GPT-4o-mini)
✔ Strict factual grounding—no hallucinations
✔ Paginated message fetching (skip/limit support)
✔ Clean JSON API output
✔ Hidden API key using .env (safe for GitHub)

🛠️ Tech Stack

FastAPI — REST API framework
Sentence Transformers (all-MiniLM-L6-v2) — semantic retrieval
OpenAI GPT-4o-mini — answer generation
python-dotenv — secure environment variable loading
Uvicorn — ASGI server

📡 API Endpoint
GET /ask
Query Parameter:
NameTypeDescriptionquestionstringNatural-language question
Example Request
GET /ask?question=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F

Example Response
{
  "answer": "Layla is planning her trip to London starting Monday for five nights."
}

Example when answer does NOT exist
{
  "answer": "I could not find that information."
}

This ensures no hallucinations — answers only use dataset content.

📥 Installation & Setup
1️⃣ Clone the repository
git clone <your-repo-url>
cd <repo-name>

2️⃣ Create a virtual environment
python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Create a .env file (DO NOT COMMIT THIS)
Add the following:
OPENAI_API_KEY=your-api-key-here

5️⃣ Run the API
cd src
uvicorn main:app --reload
The server starts at:
http://127.0.0.1:8000

🔍 How It Works
1️⃣ Fetch Messages (Paginated)
The dataset is large, so the API is fetched in batches using:
GET /messages?skip=X&limit=200
All messages are combined into one list.

2️⃣ Semantic Retrieval (Top-5)
We embed:
The user’s question
All member messages

Using:
all-MiniLM-L6-v2

Then compute cosine similarity and pick the top 5 closest messages.

3️⃣ LLM Summarization
We pass the question + retrieved messages to GPT-4o-mini with strict instructions:

Answer ONLY from these messages.
If the answer is missing, say:
“I could not find that information.”

This prevents hallucinations.

🔮 Bonus: Design Notes
1. Baseline Approaches Considered
ApproachProsConsKeyword SearchSimpleFails on natural questions; no rankingRule-based ParsingDeterministicNot scalable; brittleDense Embeddings (MiniLM)Best semantic matchNeeds vector computationHybrid Dense + Sparse (Future)Great for long docsUnnecessary for small text
Chosen:
➡️ Dense semantic search with LLM answer extraction
Because it handles paraphrasing, ambiguity, and real QA patterns.

2. Why LLM Summarization?

Ensures the final answer is short, factual, and direct Can combine multiple retrieved messages and Can gracefully say “not found” when needed, This matches real-world RAG systems.

3. Why only top-5 retrieved messages?

Keeps prompt small
Reduces noise
Still gives enough context

Bonus: Data Insights
After reviewing message data:
✔ Valid members include:
Layla Kawaguchi
Vikram Desai
Sophia Al-Farsi
Lorenzo Cavalli

etc.
“Amira” does NOT exist in the dataset
The closest name is Amina, but she has no restaurant-related preferences.
No message describes:

How many cars Vikram owns

Anyone’s "favorite restaurants"

Vehicle ownership for any user

This is why the system returns:
"I could not find that information."

This is expected and correct behavior.

🧪 Example Queries to Test
Layla trip
http://127.0.0.1:8000/ask?question=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F

Vikram cars
http://127.0.0.1:8000/ask?question=How%20many%20cars%20does%20Vikram%20Desai%20have%3F

Amira restaurants
http://127.0.0.1:8000/ask?question=What%20are%20Amira%27s%20favorite%20restaurants%3F


Alternative Approaches Considered
In addition to the final retrieval-augmented generation (semantic search + LLM summarization), the following approaches were evaluated:
1. Zero-Shot LLM Answering (No Retrieval)
This method feeds all messages—or a compressed version—directly into a large language model and asks it to answer questions based strictly on the content.
Pros: Easy to implement, handles paraphrased questions well.
Cons: Expensive, slow, not scalable, context window limitations.
2. Structured Indexing with Keyword + Metadata Filtering
This approach builds a structured inverted index using message metadata and keyword rules. Questions are answered by filtering messages by user, intent, and topic.
Pros: Fast, deterministic, cost-effective, no embeddings required.
Cons: Limited flexibility, brittle with natural-language variation.
3. Hybrid BM25 + Dense Embedding Retrieval
Combines keyword search (BM25) with semantic similarity scoring. Often produces the highest recall and precision.
Pros: High accuracy and robustness.
Cons: More complex tuning required; unnecessary for smaller datasets.

🙌 Author
Mukesh Reddy
Generative AI Engineer
Expert in RAG, FastAPI, and scalable AI pipelines.