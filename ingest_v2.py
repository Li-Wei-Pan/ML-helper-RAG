
import os
import psycopg2
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from utils import sentence_aware_chunker




OPENAI_KEY = os.getenv('OPEN_AI_KEY')
CONNECTION_STRING = os.getenv('CONNECTION_KEY')
collection_name = 'ML knowledge'

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

embeddings_v2 = OpenAIEmbeddings(
    openai_api_key=OPENAI_KEY,
    model="text-embedding-3-small"
)
collection_name = 'ML knowledge v2'


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


def ingest_v2(pdf_path, collection_name):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    chunked_docs = []
    for doc in docs:
        chunks = sentence_aware_chunker(doc.page_content, max_tokens= 150, overlap_sentence= 2)
        for chunk in chunks:
            chunked_docs.append(Document(
            page_content=chunk,
            metadata={"page": doc.metadata.get("page"), "source": doc.metadata.get("source")}
        ))

    if collection_has_data(CONNECTION_STRING, collection_name):
        print(f'collection already exist -connecting to existing data')
        db_v2 = PGVector(
        embeddings=embeddings,
        #documents=docs,
        collection_name= collection_name,
        connection =CONNECTION_STRING,
        )
    
    else:
        print(f'collection empty, embedding and inserting documents')
        db_v2 = PGVector.from_documents(
        embedding=embeddings,
        documents=chunked_docs,
        collection_name= collection_name,
        connection =CONNECTION_STRING,
    )
        print(f"Successfully stored {len(chunked_docs)} chunks in PostgreSQL.")



pdf_path = r'documents\AAAMLP.pdf'
if __name__ == "__main__":
    ingest_v2(pdf_path, collection_name)
