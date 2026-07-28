from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    response = client.get('/health')
    assert response.status_code in [200,503]
    assert 'status' in response.json() or 'detail' in response.json()

def test_empty_question():
    response = client.post('/query', json = {'question': ""})

    assert response.status_code == 422


def test_valid_question():
    response = client.post('/query', json = {'question': "Whats overfitting"})
    assert response.status_code in [200,503]


def test_odd_question():
    response = client.post('/query', json = {'question': "Is paris the capital of France"})
    assert response.status_code in [200,503]
