from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from  typing import Optional, List
from langsmith import uuid7
import json
import time
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from openai import RateLimitError, APIError
from utils import embed_single, build_context_with_budget,get_section_history, get_openai_callback, pre_flight_check,cross_reranking ,num_tokens_from_strings, api_log_request,estimate_cost,log_rejected_request, get_cross_encoder
from langchain_openai import ChatOpenAI
from contextlib import asynccontextmanager
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from sqlalchemy.ext.asyncio import create_async_engine
from pydantic import Field
from datetime import datetime
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
import sqlalchemy

OPENAI_KEY = os.getenv('OPEN_AI_KEY')
CONNECTION_STRING =os.getenv('CONNECTION_KEY_ASYNC')
MAX_CONTEXT_TOKENS = 2000

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
model = ChatOpenAI(
    model="gpt-3.5-turbo", openai_api_key=OPENAI_KEY, temperature=0)

@asynccontextmanager
async def lifespan(app: FastAPI):
        global db
        print(f"Original: {CONNECTION_STRING[:50]}")
        async_connection = CONNECTION_STRING.replace("postgresql://",
                                                     "postgresql+asyncpg://")
        
    
        async_engine = create_async_engine(async_connection, connect_args = {'options': "-c hnsw.ef_search=40"})
  
        async with async_engine.connect() as conn:
            await conn.execute(sqlalchemy.text('select 1'))

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
    top_k : int = Field(default = 10, ge = 1, le = 20)
    session_id: str = Field(default_factory= lambda: str(uuid7()))
    threshold: float = Field(default = 0.25, ge = 0.0, le = 1.0)
    use_reranking: bool = False

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
async def health():
    if db is None:
        raise HTTPException(status_code = 503, detail = 'Database not ready')
    try:
        async with db._async_engine.connect() as conn:
            await conn.execute(sqlalchemy.text('select 1'))
        return {'status': 'ok', "database": 'connected'}
    except Exception as e:
        raise HTTPException(status_code= 503, detail = f'Database unreachable : {str(e)}')


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


# Initialize limiter
limiter = Limiter(key_func= get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

#apply to endpoint
@app.post('/query', response_model= QueryResponse)
@limiter.limit('10/minute') 
async def query_endpoint(request: Request, body: QueryRequest):
    start_time = time.time()
    # Step 1 — validate input    
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")
    if db is None:
         raise HTTPException(status_code=503, detail = 'Database not initialized')
    # Step 2 — embed the question
    try:

        embedded = await embed_single(embeddings, body.question)
    
    except RateLimitError:
       raise HTTPException(status_code= 429, detail = 'embedding service rate limit reached')
    except APIError:
        raise HTTPException(status_code=503, detail="Embedding service unavailable")  
    
    # Step 3 — retrieve chunks from pgvector
    search_results = await db.asimilarity_search_with_score_by_vector(embedded, k = body.top_k)
    
    # async with db._async_engine.connect() as conn: #check if it works
    #     result = await conn.execute(sqlalchemy.text("SET hnsw.ef_search = 40"))
    #     row = await conn.execute(sqlalchemy.text("SHOW hnsw.ef_search"))
    # print(f"ef_search: {row.fetchone()[0]}")
    passing = [(doc, score) for doc, score in search_results if score <= body.threshold] # <= bc of cos distance
    print(f"Before reranking: {len(passing)} chunks")
    
    if body.use_reranking:
        passing = cross_reranking(get_cross_encoder(), body.question, passing)
        passing = passing[:body.top_k]
        context_text = "\n\n".join([doc.page_content for doc, _ in passing])
        print(f"After reranking: {len(passing)} chunks")
        for doc, score in passing:
            print(f"Reranker score: {score:.4f} | Page: {doc.metadata.get('page')}")
    
    else:
        budget_result = build_context_with_budget(passing, threshold=body.threshold, max_tokens=2000)
        context_text = "\n\n".join([doc.page_content for doc, _ in budget_result])
    # context_text = "\n\n".join([doc.page_content for doc, _ in passing])
    print(f"Context length: {len(context_text)}")
    # Step 4 — build context
    
    
    print(f"Passing chunks: {len(passing)}, threshold: {body.threshold}")
    if not passing:
         duration = time.time() - start_time
         log_rejected_request(body, duration, reason = 'no relevant chunk found')
         return QueryResponse(answer = 'NO relevant content found for your question', topic = 'unknown', is_in_document= False, sources = [])
    
    context = build_context_with_budget(passing, threshold = body.threshold, max_tokens = MAX_CONTEXT_TOKENS) 
    extract_context = [doc for doc, _ in context]
    context_text = "\n\n".join([doc.page_content for doc in extract_context])

    # Step 5 — call LLM
    
    # Step 6 — return structured response
    if len(context_text.strip()) < 50:
         return QueryResponse(
        answer='This topic is not covered in the provided material.',
        topic='unknown', 
        is_in_document=False,
        sources=[]
    )
         
    
    try:
        result = await asyncio.wait_for(chain.ainvoke({'question': body.question, 'context': context_text},config={"configurable": {"session_id": body.session_id}}), timeout= 30)
    
    except asyncio.TimeoutError:
         raise HTTPException(status_code= 503, detail = 'Request timed out')
    
    duration = time.time() - start_time
    api_log_request(body, result,duration, context_text)
    

    return QueryResponse(
        answer= result.answer,
        topic=result.topic,
        is_in_document=result.is_in_document,
        sources=[{
            "page": doc.metadata.get("page"),
            "score": float(round(score, 4)),
            "snippet": doc.page_content[:100]
        } for doc, score in passing]
    )


