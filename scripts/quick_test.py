import psycopg2
import os 
from dotenv import load_dotenv
from utils import sentence_aware_chunker
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
CONNECTION_STRING = os.getenv('CONNECTION_KEY')
collection_name = 'ML knowledge v2'

conn= psycopg2.connect(CONNECTION_STRING)
cur = conn.cursor()
cur.execute(
    """
    SELECT grantee, table_name, privilege_type
    FROM information_schema.role_table_grants
    WHERE table_name IN ('langchain_pg_embedding', 'langchain_pg_collection')
"""
)

# cur.execute(
#     """SELECT
#     kcu.table_name AS student_table,
#     kcu.column_name AS foreign_key_column,
#     ccs.table_name AS parent_table,
#     ccs.column_name AS primary_key_column
# FROM information_schema.table_constraints AS tc
# JOIN information_schema.key_column_usage AS kcu
#   ON tc.constraint_name = kcu.constraint_name
# JOIN information_schema.constraint_column_usage AS ccs
#   ON ccs.constraint_name = tc.constraint_name
# WHERE tc.constraint_type = 'FOREIGN KEY'
#   AND kcu.table_name = 'langchain_pg_embedding';"""
# )
# print(f"Chunks in v2: {cur.fetchone()[0]}")
conn.close()



# loader = PyPDFLoader(r'documents\AAAMLP.pdf')
# docs = loader.load()

# check a sample page
# sample_page = docs[19].page_content  # page 19 had overfitting content
# chunks = sentence_aware_chunker(sample_page, max_tokens=500, overlap_sentence=2)
# print(f"Page 19 token estimate: ~{len(sample_page.split())} words")
# print(f"Chunks produced: {len(chunks)}")
# for i, c in enumerate(chunks):
#     print(f"Chunk {i+1}: {c[:100]}")