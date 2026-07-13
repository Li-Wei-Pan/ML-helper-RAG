# utils.py
import tiktoken
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import json
import os
from datetime import datetime
import sqlalchemy

MAX_CONTEXT_TOKENS = 2000

def num_tokens_from_strings(string, encoding_name):
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        print('num_token used for question', num_tokens)
        return num_tokens
    except KeyError:
        print('wrong encoding model')

def pre_flight_check(question, model="cl100k_base", max_tokens=MAX_CONTEXT_TOKENS):
    token_count = num_tokens_from_strings(question, model)
    if token_count > max_tokens:
        return False
    return True

def build_context_with_budget(results, threshold=0.3, max_tokens=MAX_CONTEXT_TOKENS):
    context = ""
    total_tokens = 0
    enc = tiktoken.get_encoding("cl100k_base")
    for doc, score in results:
        chunk_tokens = len(enc.encode(doc.page_content))
        if total_tokens + chunk_tokens > max_tokens:
            break
        context += doc.page_content + "\n\n"
        total_tokens += chunk_tokens
    return context

store = {}
def get_section_history(session_id) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

async def embed_single(embeddings_model, text):
    return await embeddings_model.aembed_query(text)

def api_log_request(body, result, duration, context):
    input_tokens  = num_tokens_from_strings(context + body.question, "cl100k_base")
    output_tokens = num_tokens_from_strings(result.answer, "cl100k_base")
    cost          = estimate_cost(input_tokens, output_tokens)
    log_entry = {
        'timestamp':        datetime.utcnow().isoformat(),
        'session_id':       body.session_id,
        'question':         body.question,
        'answer':           result.answer,
        'topic':            result.topic,
        'is_in_document':   result.is_in_document,
        'threshold':        body.threshold,
        'top_k':            body.top_k,
        'duration_seconds': round(duration, 3),
        'input_tokens':     input_tokens,
        'output_tokens':    output_tokens,
        'cost_usd':         cost
    }

    with open('api_log.jsonl', 'a')as f:
        f.write(json.dumps(log_entry)+ "\n")

def log_rejected_request(body, duration, reason):
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'session_id': body.session_id,
        'question': body.question,
        'answer': None,
        'topic': None,
        'is_in_document': False,
        'threshold': body.threshold,
        'top_k': body.top_k,
        'duration_seconds': round(duration, 3),
        'input_tokens': 0,
        'output_tokens': 0,
        'cost_usd': 0,
        'rejection_reason': reason
    }
    with open('api_log.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost  = (input_tokens  / 1000) * 0.0005
    output_cost = (output_tokens / 1000) * 0.0015
    return round(input_cost + output_cost, 6)


async def push_and_query_db_query(vector_store, texts, embeddings):
    print("Starting async database insertion...")
    await vector_store.aadd_texts(texts = texts)
    print(f'successfully inserted into pgvector async')

    query = 'what is overfitting'
    print(f'seraching {query} async...')
    query_vector = await embeddings.aembed_query(query)
    results = await vector_store.asimilarity_search_by_vector(query_vector, k = 2)
    return results