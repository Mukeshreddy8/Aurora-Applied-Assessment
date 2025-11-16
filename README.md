Member Question Answering API
Overview
This project implements a production-ready Question Answering API that answers natural-language questions about member messages.
It uses semantic search to retrieve the closest messages and an LLM to generate a grounded answer based strictly on the retrieved text.
If the information is not present in the dataset, the API returns:
“I could not find that information.”
This fully satisfies the assignment requirement:
The answer must be inferred only from the member messages provided by the public API.
Features
• Semantic retrieval using OpenAI embeddings
• Top-1 relevant message matching using cosine similarity
• LLM-based answer generation using GPT-4o-mini
• Strict grounding — no hallucinated answers
• FastAPI backend deployed publicly
• Environment variables used for API key security
• Clean JSON output format
• Paginated dataset fetching
Public API Endpoint
Base URL:
https://aurora-applied-assessment.onrender.com
Endpoint:
GET /ask
Query parameter:
• question → natural-language question
Example Queries (Live Endpoints)
Layla – London trip
https://aurora-applied-assessment.onrender.com/ask?question=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F
Vikram – car ownership
https://aurora-applied-assessment.onrender.com/ask?question=How%20many%20cars%20does%20Vikram%20Desai%20have%3F
Amira – favorite restaurants
https://aurora-applied-assessment.onrender.com/ask?question=What%20are%20Amira%27s%20favorite%20restaurants%3F
Installation and Local Setup
Clone the repository
Create a virtual environment (python3 -m venv venv)
Activate it (source venv/bin/activate)
Install dependencies (pip install -r requirements.txt)
Create a .env file and add your OpenAI API key
Start the server by running:
uvicorn src.main:app --reload
Local server runs at: http://127.0.0.1:8000
How the System Works
Data Fetching
The system fetches member messages using the public endpoint: GET /messages
Messages are fetched in batches and combined into one complete dataset.
Semantic Search
The system embeds the user question and all message texts using OpenAI’s “text-embedding-3-small”.
Cosine similarity is computed, and the most relevant message is selected.
LLM Answer Extraction
GPT-4o-mini is used to answer the question based only on the retrieved message.
If the message does not contain the answer, the system replies with:
“I could not find that information.”
This ensures the model never invents information.
Dataset Insights
• Layla Kawaguchi has multiple records including her London trip.
• There is no information about how many cars Vikram Desai owns.
• “Amira” does not exist in the dataset — only “Amina”, and she has no restaurant preferences.
• No user mentions favorite restaurants explicitly.
Therefore:
• Layla question → answerable
• Vikram and Amira questions → “I could not find that information.”
This behavior is correct and matches real-world QA constraints.
Alternative Approaches Considered
Keyword-Based Search
Pros: Simple
Cons: Fails with natural language wording
Rule-Based Message Parsing
Pros: Fast
Cons: Not scalable, breaks easily
Zero-Shot LLM Answering Without Retrieval
Pros: Easy
Cons: Expensive and prone to hallucinations
Hybrid BM25 + Dense Vector Retrieval
Pros: Better recall
Cons: Not required for small text snippets
Final choice: Dense semantic retrieval + LLM summarization
This offers the best balance of accuracy, reliability, cost, and simplicity.
Final Output Format
Successful answer example:
“Layla is planning her trip to London starting Monday for five nights.”
Missing information example:
“I could not find that information.”
Author
Mukesh Reddy
Generative AI Engineer with expertise in RAG, FastAPI, and production-grade AI systems.
Cons: More complex tuning required; unnecessary for smaller datasets.

🙌 Author
Mukesh Reddy
Generative AI Engineer
Expert in RAG, FastAPI, and scalable AI pipelines.
