from fastapi.testclient import TestClient
from api import app, verify_api_key
import pytest

client = TestClient(app)

def override_verify_api_key():
    return "test-key"

app.dependency_overrides[verify_api_key] = override_verify_api_key

ML_TEST_CASES = [
    ("What is cross-validation?", ["training", "fold", "generalization"], True),
    ("What is overfitting?", ["training", "test", "generalize"], True),
    ("What is AUC?", ["ROC", "curve", "classification"], True),
    ("What is the capital of France?", [], False),  # OOD
]

def test_health():
    response = client.get('/health')
    assert response.status_code in [200, 503]
    assert 'status' in response.json() or 'detail' in response.json()

def test_empty_question():
    response = client.post('/query', json={'question': ""})
    assert response.status_code == 422

def test_valid_question():
    response = client.post('/query', json={'question': "Whats overfitting"})
    assert response.status_code in [200, 503]

def test_odd_question():
    response = client.post('/query', json={'question': "Is paris the capital of France"})
    assert response.status_code in [200, 503]

@pytest.mark.parametrize("question,keywords, expected_in_doc", ML_TEST_CASES)
def test_ml_concepts(question, keywords, expected_in_doc):
    response = client.post('/query', json={'question': question})
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data['is_in_document'] == expected_in_doc
        answer_lower = data['answer'].lower()
        if expected_in_doc and keywords:
            assert any(kw.lower() in answer_lower for kw in keywords)