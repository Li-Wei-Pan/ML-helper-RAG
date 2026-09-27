# ML Helper RAG

A production-grade RAG system that answers ML questions using content from "Approaching Almost Any Machine Learning Problem" by Abhishek Thakur.

## What it does
It's a RAG driven assistant that can solve Machine-Learning related questions and theories as long as the answers are present in the book. 

## Architecture

Powered by Postgresql's PGVector database to store embeeded chunks and using OpenAI's model as main LLM model. It checks user query's syntax and length to check if it meets the requirements(safety and prevent prompt injection). Retrieval uses pgvector's HNSW index (m=16, ef_search=40) for approximate nearest neighbor search, with cosine distance threshold filtering and optional cross-encoder reranking.

The orchestrator agent is ReAct and can assign works to three agents(ML, calculator, summary agent) based on user query. The agent will iteratively updating/taking actions based on its current knowledge(INPUT, ACTION). 
INPUT: the content of user query
ACTION: the tool it chose to use (ml, math, summary)

The agent will return result when it reaches the FINAL state.

## Tech Stack
Database: PGVector
OpenAI ada-002 embeddings, turbo 3.5
sentence-transformer(cross-encoder)
Pydantic
Agent orchestrator: machine_learning agent, calculator_agent, summary_agent
FastAPI
Docker
CI/CD
Github actions
Python 3.12
UV

## How to Run
### Prerequisites
- Docker Desktop
- OpenAI API key
- `.env` file with credentials (see `.env.example`)

### Start the system
```bash
docker compose -f docker/docker-compose.yml up
```

### API is available at
- `http://localhost:8000/health`
- `http://localhost:8000/docs` (Swagger UI)

### Stop
```bash
docker compose -f docker/docker-compose.yml down
```

## .env example
OPEN_AI_KEY=your_openai_key_here
CONNECTION_KEY=postgresql://user:password@localhost:5432/dbname
CONNECTION_KEY_ASYNC=postgresql+asyncpg://user:password@localhost:5432/dbname
POSTGRES_DB=vector_db
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password

## API Endpoints

### `GET /health`
Returns database connection status.

### `POST /query`
RAG retrieval + LLM generation.
```json
{
  "question": "What is cross-validation?",
  "top_k": 10,
  "threshold": 0.25,
  "use_reranking": false
}
```

### `POST /agent`
ReAct agent with tool orchestration.
```json
{
  "question": "What is the difference between bagging and boosting?"
}
```


## Evaluation

Evaluated against a 100-question golden dataset generated from the source document.

| Metric | Score |
|--------|-------|
| Recall@k=4 | 64% |
| Recall@k=10 | 72% |
| Avg query latency | 0.65s |
| Threshold failures | 0 |
| Retrieval failures (not in top-20) | 25% |

Cross-encoder reranking tested but showed no recall improvement on current dataset — disabled by default (`use_reranking: false`).
**Known limitation:** 25% of questions are not retrievable within top-20 results. Root cause is character-based chunking splitting semantic units across chunk boundaries. Sentence-aware chunking (`ingest_v2.py`) was tested but showed lower recall on the current golden dataset due to evaluation bias — the dataset was generated from the original chunks. A human-labeled evaluation dataset would give a fairer comparison.


## Project Structure
├── api.py          # FastAPI backend, /query and /agent endpoints
├── agent.py        # ReAct agent with tool orchestration
├── utils.py        # shared utilities — chunking, logging, evaluation
├── ingest_v2.py    # document ingestion pipeline
├── app.py          # Streamlit UI (alternative interface)
├── docker/         # Dockerfile and docker-compose.yml
├── scripts/        # evaluation and testing scripts
├── notebooks/      # development notebooks
└── tests/          # pytest test suite