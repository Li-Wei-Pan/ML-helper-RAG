"""
Seeds a handful of hardcoded ML knowledge chunks into the test database.

Used by CI to populate an ephemeral Postgres+pgvector instance so that
tests/test_api.py's retrieval-dependent assertions have real content to
match against. This is NOT the production ingestion pipeline — that lives
in the notebook and loads the actual book — this is just enough content
for the specific questions in ML_TEST_CASES to be answerable.
"""
import asyncio
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

load_dotenv()

OPENAI_KEY = os.getenv('OPEN_AI_KEY')
CONNECTION_STRING = os.getenv('CONNECTION_KEY_ASYNC') or os.getenv('CONNECTION_KEY')
COLLECTION_NAME = 'ML knowledge'

SEED_CHUNKS = [
    {
        'text': (
            "Cross-validation is a technique for evaluating how well a model "
            "generalizes to unseen data. The most common form, k-fold cross-validation, "
            "splits the training data into k folds; the model is trained on k-1 folds "
            "and validated on the remaining fold, rotating through all folds so every "
            "example is used for both training and validation exactly once."
        ),
        'page': 21,
    },
    {
        'text': (
            "Overfitting occurs when a model learns patterns specific to the training "
            "data, including noise, rather than the underlying signal that generalizes "
            "to new data. A model that performs very well on training data but poorly "
            "on test data is a classic sign of overfitting, usually meaning the model "
            "is too complex relative to the amount of training data available."
        ),
        'page': 20,
    },
    {
        'text': (
            "AUC, or area under the curve, refers to the area under the ROC (Receiver "
            "Operating Characteristic) curve, which plots the true positive rate against "
            "the false positive rate at various threshold settings. AUC is a widely used "
            "metric for evaluating binary classification models, with a value of 1.0 "
            "indicating a perfect classifier and 0.5 indicating performance no better "
            "than random guessing."
        ),
        'page': 30,
    },
    {
        'text': (
            "Stratified k-fold cross-validation preserves the percentage of samples for "
            "each class in every fold. This is especially important for imbalanced "
            "classification problems, where a regular k-fold split could produce folds "
            "with very few or no examples of the minority class."
        ),
        'page': 22,
    },
    {
        'text': (
            "Supervised learning is a machine learning paradigm where a model is trained "
            "on labeled data — pairs of inputs and their corresponding correct outputs. "
            "The model learns a mapping from inputs to outputs during training, and its "
            "goal is to generalize that mapping to make accurate predictions on new, "
            "unseen inputs."
        ),
        'page': 7,
    },
]


def ensure_extension(sync_connection_string: str) -> None:
    # single-statement, sync — avoids the asyncpg multi-statement error
    sync_engine = create_engine(sync_connection_string)
    with sync_engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    sync_engine.dispose()


async def seed() -> None:
    ensure_extension(CONNECTION_STRING)

    async_connection = CONNECTION_STRING.replace("postgresql://", "postgresql+asyncpg://")
    async_engine = create_async_engine(async_connection)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    db = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=async_engine,
        create_extension=False,
    )

    texts = [chunk['text'] for chunk in SEED_CHUNKS]
    metadatas = [{'page': chunk['page']} for chunk in SEED_CHUNKS]

    await db.aadd_texts(texts=texts, metadatas=metadatas)
    print(f"Seeded {len(texts)} chunks into collection '{COLLECTION_NAME}'.")

    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())