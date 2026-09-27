import sqlalchemy
import json
import os
import time
import sys 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pydantic import Field, BaseModel
from langchain_postgres import PGVector
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
from utils import cross_reranking, get_cross_encoder, evalute_with_reranking, fetch_random_chunks,generate_questions_from_chunks,validate_question,build_golden_dataset
from sqlalchemy import create_engine


load_dotenv()
OPENAI_KEY = os.getenv('OPEN_AI_KEY')
CONNECTION_STRING = os.getenv('CONNECTION_KEY')
emb_model = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=OPENAI_KEY,
    temperature=0
)

sync_engine = create_engine(CONNECTION_STRING)
db = PGVector(
    embeddings=emb_model,
    collection_name='ML knowledge',
    connection=sync_engine  # test
)

db_v2 = PGVector(
    embeddings=emb_model,
    collection_name='ML knowledge v2',
    connection=sync_engine
)

client = OpenAI(api_key=OPENAI_KEY)

class JudgeOutput(BaseModel):
    faithfulness: int = Field(..., description="Score 1-5 verifying claims against context")
    relevance: int = Field(..., description="Score 1-5 checking if answer addresses query")
    hallucination: int = Field(..., description="Score 1-5 where 5 means NO hallucination, 1 means heavy hallucination")
    reasoning: str = Field(..., description="One sentence explaining the scores")



def diagnose_recall_failures(golden_dataset, embedding_model, db, top_k=10, threshold=0.25):
    threshold_failures = 0
    retrieval_failures = 0
    
    for item in golden_dataset:
        embedded_q = embedding_model.embed_query(item['question'])
        # search with loose threshold to see if chunk exists at all
        raw_results = db.similarity_search_with_score_by_vector(embedded_q, 20)
        
        all_pages = [doc.metadata.get('page') for doc, score in raw_results]
        passing_pages = [doc.metadata.get('page') for doc, score in raw_results if score <= threshold]
        
        if item['expected_page'] not in passing_pages:
            if item['expected_page'] in all_pages:
                threshold_failures += 1  # chunk found but filtered out
            else:
                retrieval_failures += 1  # chunk not found at all in top-20
    
    print(f"Threshold failures: {threshold_failures}")
    print(f"Retrieval failures (chunk not in top-20): {retrieval_failures}")


def run_evaluation():
    start = time.time()
    #expanded_dataset = build_golden_dataset(db, emb_model, sync_engine, 25, 2) #create dataset it not exist
    with open('golden_dataset.json', 'r') as f:
        expanded_dataset = json.load(f)


    print(f"Total questions: {len(expanded_dataset)}")
    #result = evalute_with_reranking(expanded_dataset[:5], emb_model, db, get_cross_encoder(), top_k= 10, threshold= 0.25)
#     result_v2 = evalute_with_reranking(
#     expanded_dataset, emb_model, db_v2, 
#     get_cross_encoder(), top_k=10, threshold=0.25
# )
    result_4 = evalute_with_reranking(expanded_dataset[:100], emb_model, db, get_cross_encoder(), top_k=4, threshold=0.25)
    result_10 = evalute_with_reranking(expanded_dataset[:100], emb_model, db, get_cross_encoder(), top_k=10, threshold=0.25)

    diagnose_recall_failures(expanded_dataset[:100], emb_model, db, top_k=10, threshold=0.25)
    duration = time.time() - start
    
    print("=== RAG Evaluation Report ===")
    print(f"Dataset: {len(expanded_dataset[:100])} questions")
    #print(f"result :  {result}")
    print(f"top_k=4:  {result_4}")
    print(f"top_k=10: {result_10}")
    print(f"Total eval time: {duration:.1f}s")
    print(f"Avg per query: {duration/len(expanded_dataset[:100]):.3f}s")
    print("=============================")

if __name__ == '__main__':
    run_evaluation()