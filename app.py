import streamlit as st
import os
import psycopg2
from dotenv import load_dotenv
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool


load_dotenv()
st.set_page_config(page_title= "ML-RAG Helper", layout = 'wide')
st.title('ML helper')
st.caption('Ask any ML-related problem...')

def collection_has_data(connection_str, collection_name):
    try:

        conn = psycopg2.connect(connection_str)
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            SELECT COUNT(*) FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s
        """, (collection_name,))
        count = cur.fetchone()[0]
        conn.close()
        return count > 0
    except:
        return False

store = {}
def get_section_history(session_id) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

@st.cache_resource
def load_resources():
    connection_str = os.getenv("CONNECTION_KEY")
    collection_name = 'AAAMLP.pdf'
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPEN_AI_KEY'))

    
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv('OPEN_AI_KEY'), temperature=0)
    db = PGVector(
        embeddings=embeddings,
        collection_name="document1",
        connection=connection_str,
    )
    return embeddings, model, db

embeddings, model, db = load_resources()

# Prompt
template = """You are a helpful assistant. You will be provided with a document and a question.
Your task is to answer the question based on the content of the document. 
here are the relevant section: {context}
Question: {question}
These are the past messages between you and the user, please use this history list to answer the
question as well.
History: {history}"""


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful ML assistant. Context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | model
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_section_history,
    input_messages_key = 'question',
    history_messages_key = 'history'
)

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'history' not in st.session_state:
    st.session_state.history = []

# side bar
with st.sidebar:
    st.header('Settings')
    k = st.slider('Chunks to retrieve (k)', 2,8,4)
    show_sources = st.toggle('Show source chunks', value= True)

    if st.button('Clear chat'):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()


#render past msg
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if show_sources and msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])} chunks)"):
                for i, doc in enumerate(msg["sources"]):
                    page = doc.metadata.get("page", "?")
                    st.caption(f"Chunk {i+1} — page {page}")
                    st.write(doc.page_content[:300] + "...")

# input
if question := st.chat_input('Ask a question about your book...'):
    st.session_state.messages.append({'Role': 'user', 'content': question})
    with st.chat_message('user'):
        st.write(question)
    
    with st.chat_message('assistant'):
        with st.spinner('Searching....'):
            embedded_question = embeddings.embed_query(question)
            context = db.similarity_search_by_vector(embedded_question, k = k)
            result = chain_with_history.invoke(
        {"question": question, "context": context}, 
        config={"configurable": {"session_id": "user_123"}}
    )
            answer = result.content 
        st.write(answer)
    
        if show_sources:
            with st.expander(f"Sources ({len(context)} chunks)"):
                for i, doc in enumerate(context):
                    page = doc.metadata.get("page", "?")
                    st.caption(f"Chunk {i+1} — page {page}")
                    st.write(doc.page_content[:300] + "...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": context
    })
    st.session_state.history.append(
        f"Human: {question} | Assistant: {answer}"
    )