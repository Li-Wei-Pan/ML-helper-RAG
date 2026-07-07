from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from  typing import Optional, List
from langsmith import uuid7
import json
import time
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from utils import embed_single, build_context_with_budget,get_section_history, get_openai_callback, pre_flight_check, num_tokens_from_strings
from langchain_openai import ChatOpenAI
from contextlib import asynccontextmanager
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from sqlalchemy.ext.asyncio import create_async_engine
from pydantic import Field
from datetime import datetime

OPENAI_KEY = os.getenv('OPEN_AI_KEY')
CONNECTION_STRING =os.getenv('CONNECTION_KEY_ASYNC')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
model = ChatOpenAI(
    model="gpt-3.5-turbo", openai_api_key=OPENAI_KEY, temperature=0)

@asynccontextmanager
async def lifespan(app: FastAPI):
        global db
        print(f"Original: {CONNECTION_STRING[:50]}")
        async_connection = CONNECTION_STRING.replace("postgresql://",
                                                     "postgresql+asyncpg://")
        async_engine = create_async_engine(async_connection)
        print(f"Async: {async_connection[:60]}")
        db = PGVector(
            embeddings=embeddings,
            collection_name="ML knowledge",
            connection=async_engine,
        )
        print('Database connection established!')
        yield
        await async_engine.dispose()
        print('Shutting down')




# embeddings_v2 = OpenAIEmbeddings(
#     openai_api_key=OPENAI_KEY,
#     model="text-embedding-3-small"
# )
db = None
app = FastAPI(lifespan= lifespan)

# request schemea
class QueryRequest(BaseModel):
    question: str
    top_k : int = Field(default = 4, ge = 1, le = 20)
    session_id: str = Field(default_factory= lambda: str(uuid7()))
    threshold: float = Field(default = 0.25, ge = 0.0, le = 1.0)

# source chunk schema
class SourceChunk(BaseModel):
    page: int
    score: float
    snippet: str

# response schema
class QueryResponse(BaseModel):
    answer: str
    topic: str
    is_in_document: bool
    sources: list

class LLMOutputSchema(BaseModel):
    topic: str
    is_in_document: bool
    source_chunks_used: List[str]
    answer: str

@app.get('/health')
def health():
    return {'status': 'ok'}

section_id = str(uuid7())
prompt = ChatPromptTemplate.from_messages([ 
     ("system", """You are a helpful ML assistant.

Use ONLY the following retrieved context to answer the question.
If the answer cannot be found in the context, you MUST:
- Set is_in_document to false
- State that the topic is not covered in the provided material
- Do NOT answer from general knowledge

Context: {context}

You must respond with the following fields:
- topic: the main ML topic the question is about
- is_in_document: true only if the answer is found in the context above
- source_chunks_used: list of direct quotes from the context you used
- answer: your answer based solely on the context provided
"""),
    #MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

structured_model = model.with_structured_output(LLMOutputSchema, method="function_calling")
chain = prompt | structured_model
# chain_with_history = RunnableWithMessageHistory(
#     chain,
#     get_section_history,
#     input_messages_key='question',
#     history_messages_key='history'
# )

@app.post('/query', response_model= QueryResponse)
async def query_endpoint(request: QueryRequest):
    # Step 1 — validate input    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")
    if db is None:
         raise HTTPException(status_code=503, detail = 'Database not initialized')
    # Step 2 — embed the question
    embedded = await embed_single(embeddings, request.question)
    # Step 3 — retrieve chunks from pgvector
    search_results = await db.asimilarity_search_with_score_by_vector(embedded, k = request.top_k)
    # Step 4 — build context
    passing = [(doc, score) for doc, score in search_results if score <= request.threshold] # <= bc of cos distance
    print(f"Passing chunks: {len(passing)}, threshold: {request.threshold}")
    if not passing:
         return QueryResponse(answer = 'NO relevant content found for your question', topic = 'unknown', is_in_document= False, sources = [])
    context = build_context_with_budget(passing, threshold = request.threshold)
    # Step 5 — call LLM
    
    # Step 6 — return structured response
    if len(context.strip()) < 50:
         return QueryResponse(
        answer='This topic is not covered in the provided material.',
        topic='unknown', 
        is_in_document=False,
        sources=[]
    )
         
    start_time = time.time()
    result = chain.invoke({'question': request.question, 'context': context},config={"configurable": {"session_id": section_id}})
    duration = time.time() - start_time

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": request.session_id,
        "question": request.question,
        "answer": result.answer,
        "topic": result.topic,
        "is_in_document": result.is_in_document,
        "threshold": request.threshold,
        "top_k": request.top_k,
        "duration_seconds": round(duration, 3),
        "context_chars": len(context)
    }

    with open('user_api_log.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')



    return QueryResponse(
        answer= result.answer,
        topic=result.topic,
        is_in_document=result.is_in_document,
        sources=[{
            "page": doc.metadata.get("page"),
            "score": round(score, 4),
            "snippet": doc.page_content[:100]
        } for doc, score in passing]
    )