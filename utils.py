# utils.py
import tiktoken
import nltk
nltk.download('punkt_tab')
import json
import time
import asyncio
import os
import sqlalchemy
import random
import openai
import tiktoken
from openai import OpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from datetime import datetime
from typing import Callable, Any
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize

load_dotenv()
OPENAI_KEY = os.getenv('OPEN_AI_KEY')
CONNECTION_STRING = os.getenv('CONNECTION_KEY')
MAX_CONTEXT_TOKENS = 2000
client = OpenAI(api_key=OPENAI_KEY)
emb_model = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

def num_tokens_from_strings(string, encoding_name):
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        #print('num_token used for question', num_tokens)
        return num_tokens
    except KeyError:
        print('wrong encoding model')

def pre_flight_check(question, model="cl100k_base", max_tokens=MAX_CONTEXT_TOKENS):
    token_count = num_tokens_from_strings(question, model)
    if token_count > max_tokens:
        return False
    return True

def sentence_aware_chunker(texts, max_tokens = 500, overlap_sentence = 2, encoding_name = 'cl100k_base'):
    chunks = []
    batch_chunk = []
    sentences = sent_tokenize(texts) # to get sentences
    # enc = tiktoken.get_encoding(encoding_name) #get encoder
    # print('enc:',enc)

    for sentence in sentences:
        token_count_of_next_seq = num_tokens_from_strings(sentence, encoding_name)
        #print('token_count_of_next_seq:', token_count_of_next_seq)
        current_token_count = num_tokens_from_strings(" ".join(batch_chunk),  encoding_name)
        #print('current_token:', current_token_count)


        if current_token_count + token_count_of_next_seq <= max_tokens:
            batch_chunk.append(sentence)
            
        else: #exceed max tokens
            if batch_chunk:
                chunks.append(" ".join(batch_chunk))

            last_sentence = batch_chunk[-overlap_sentence:]
            batch_chunk = last_sentence + [sentence]
        
    if batch_chunk:
        chunks.append(" ".join(batch_chunk))  
    return chunks
            
            
sample = "Machine learning is a subset of AI. It enables computers to learn from data. Neural networks are inspired by the brain. They consist of layers of neurons. Deep learning uses many layers."
chunks = sentence_aware_chunker(sample, max_tokens=20, overlap_sentence=1)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
# sentence_aware_chunker('testing new thing')



def build_context_with_budget(raw_results, threshold, max_tokens, encoding_name = 'cl100k_base'):
    result = []
    enc = tiktoken.get_encoding(encoding_name)
    token_count = 0
    for doc, score in sorted(raw_results, key= lambda x: x[1]):
        # if score > threshold:
        #     continue
        chunk_tokens = len(enc.encode(doc.page_content))
        if chunk_tokens + token_count >  max_tokens:
            print(f'Context budget reached at {token_count} tokens - dropping remaining chunks')
            break
        result.append((doc, score))
        token_count += chunk_tokens
    print(f'Context assembled {len(result)} chunks, {token_count} tokens')
    return result

#exponential delay with jitter
async def async_exp_delay_jitter(max_attempts: int, fn: Callable, *args, **kwargs):
    result = None
    delay = 1
    last_exception = None
    for attempt in range(0, max_attempts):
        try: 
            result = await fn(*args, **kwargs)
            return result
        except Exception as e:
           last_exception = e
           if isinstance(e,openai.AuthenticationError) or isinstance(e,openai.BadRequestError):
                raise ValueError(f"Bad request or authentication issue")
           
           elif isinstance(e, openai.RateLimitError) or isinstance(e, openai.APIStatusError) and e.status_code in (500, 503):
                temp_time = delay * (2**attempt) + random.random()
                print(f"Attempt:{attempt} Error: {e} Waiting time: {temp_time}")
                await asyncio.sleep(temp_time)

           else:
               raise e
           
        
    if result is None:
        raise last_exception
        
    return result

store = {}
def get_section_history(session_id) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

async def embed_single(embeddings_model, text):
    return await embeddings_model.aembed_query(text)


async def push_and_query_db_query(vector_store, texts, embeddings):
    print("Starting async database insertion...")
    await vector_store.aadd_texts(texts = texts)
    print(f'successfully inserted into pgvector async')

    query = 'what is overfitting'
    print(f'seraching {query} async...')

    query_vector = await embeddings.aembed_query(query)
    results = await vector_store.asimilarity_search_by_vector(query_vector, k = 2)
    return results


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



cross_enc_mod = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

def cross_reranking(cross_model, query, retrieved_chunks):
    raw_pairs = [[query, chunk[0].page_content] for chunk in retrieved_chunks]
    scores = cross_model.predict(raw_pairs)
    original_docs = [chunk[0] for chunk in retrieved_chunks]
    paired_result = list(zip(original_docs, scores))
    sorted_results = sorted(paired_result, key = lambda x: x[1], reverse = True)

    return sorted_results



def evalute_with_reranking(golden_dataset, embedding_model, db, cross_model, top_k, threshold):
    vanilla_hits = 0
    cross_hits = 0

    for item in golden_dataset:
        embedded_q = embedding_model.embed_query(item['question'])
        raw_results = db.similarity_search_with_score_by_vector(embedded_q, top_k)
        filtered = [(doc,score) for doc,score in raw_results if score <= threshold]
        sorted_results = cross_reranking(cross_model, item['question'], filtered)

        vanilla_pages = [doc.metadata.get('page') for doc,score in raw_results if score <= threshold]
        reranked_pages = [doc.metadata.get('page') for doc, score in sorted_results[:top_k]]

        if item['expected_page'] in vanilla_pages:
            vanilla_hits+=1
        if item['expected_page'] in reranked_pages:
            cross_hits += 1
        
    return f"Vanilla's recall@k: {vanilla_hits} | Cross encoder's recall@k: {cross_hits}"

_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    return _cross_encoder




def fetch_random_chunks(n_question, sync_engine, collection_name):
    #b = PGVector(embeddings= emb_model, collection_name= 'ML knowledge', connection = async_engine)
    with sync_engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(
        """
        SELECT document, cmetadata
        FROM  langchain_pg_embedding
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection
            where name =:collection_name)
        AND length(document) > 250
        AND document NOT ILIKE '%Abhishek Thakur%'
        ORDER BY RANDOM()
        LIMIT :n
        """
        ), {'collection_name': collection_name, 'n': n_question})
        rows = result.fetchall()

    return [
        {
            'text': row[0],
            'page': row[1].get('page')
        }
        for row in rows
    ]

# CONNECTION_STRING = os.getenv('CONNECTION_KEY')

def generate_questions_from_chunks(chunk_text, page_num, n_questions = 2):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        temperature=0.7,
        messages=[
            {
                'role': 'system',
                'content': """You are an expert machine learning dataset generator. Given a text chunk, generate specific, highly technical questions that can ONLY be answered using the core technical concepts, algorithms, or code in that chunk.
                STRICT RULES:

                Do NOT generate questions about authors, book titles, page numbers, publication dates, or formatting.

                Focus exclusively on machine learning mechanics, equations, and data science methodologies.
                Output JSON only: {\"questions\": [\"q1\", \"q2\"]}"""
            },
            {
                'role': 'user',
                'content': f"Generate {n_questions} questions from this text:\n\n{chunk_text}"
            }
        ]
    )
    raw = response.choices[0].message.content
    prased = json.loads(raw)
    return [{'question': q ,'expected_page': page_num} for q in prased['questions']]

def validate_question(question_text, existing_question):
    if question_text in existing_question:
        return False
    forbidden_words = ['author', 'book', 'page number', 'published', 'written by']
    question_lower = question_text.lower()
    if any( term in question_lower for term in forbidden_words):
        return False
    if len(question_text) < 15:
        return False
    return True


MIN_DATASET_SIZE = 50
def build_golden_dataset(db, embeddings, engine, n_chunks = 25, question_per_chunk = 2):
    existing_questions_set = set()

    json_data = []
    if os.path.isfile('golden_dataset.json'):
        with open('golden_dataset.json', 'r') as f:
            json_data = json.load(f)
            # Extract strings into a fast O(1) lookup set
            existing_questions_set = {item['question'] for item in json_data}
            if len(json_data) >= MIN_DATASET_SIZE:
                print(f"Dataset already has {len(json_data)} questions — skipping generation")
                return json_data
    else:

        print('No existing path found, creating a new file named golden_dataset.json.')


    new_questions = []
    random_chunks = fetch_random_chunks(n_chunks, engine, 'ML knowledge')


    for chunk_text in random_chunks:
        # print(f'chunk_text: {chunk_text}')
        generated_questions = generate_questions_from_chunks(chunk_text['text'], chunk_text['page'], question_per_chunk)
        for q_dict in generated_questions:
            q_text = q_dict['question']
            if validate_question(q_text, existing_questions_set) is True:
                new_questions.append(q_dict)
                existing_questions_set.add(q_text)
        
    with open('golden_dataset.json', 'w') as f:
        json_data.extend(new_questions)
        json.dump(json_data, f, indent=2)

    return json_data
    
def evalute_with_reranking(golden_dataset, embedding_model, db, cross_model, top_k, threshold):
    vanilla_hits = 0
    cross_hits = 0

    for item in golden_dataset:
        embedded_q = embedding_model.embed_query(item['question'])
        raw_results = db.similarity_search_with_score_by_vector(embedded_q, top_k)
        filtered = [(doc,score) for doc,score in raw_results if score <= threshold]
        sorted_results = cross_reranking(cross_model, item['question'], filtered)

        vanilla_pages = [doc.metadata.get('page') for doc,score in raw_results if score <= threshold]
        reranked_pages = [doc.metadata.get('page') for doc, score in sorted_results[:top_k]]

        if item['expected_page'] in vanilla_pages:
            vanilla_hits+=1
        if item['expected_page'] in reranked_pages:
            cross_hits += 1
        
    return f"Vanilla's recall@k: {vanilla_hits } | Cross encoder's recall@k: {cross_hits}"



def log_agent_run(session_id, question, final_answer, steps_taken, tools_used, success):
    json_path = 'log_agent.json'
    log_entry = {
    'session_id': session_id,
    'question': question,
    'final_answer': final_answer,
    'steps_taken': steps_taken,
    'tools_used': [list(t) for t in tools_used],
    'success': success,
    'timestamp': datetime.utcnow().isoformat()
}
    with open(json_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    