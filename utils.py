# utils.py
import tiktoken
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


MAX_CONTEXT_TOKENS = 2000

def num_tokens_from_strings(text, model):
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))

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
        if score <= threshold:
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