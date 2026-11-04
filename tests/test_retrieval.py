import pytest
from fastapi.testclient import TestClient
from app import core
from app.main import app

@pytest.fixture(autouse=True)
def isolated_db(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'DB_PATH',str(tmp_path/'citemesh.sqlite'))
    core.initialize()

@pytest.fixture
def client():
    with TestClient(app) as c:yield c

def test_seed_and_grounded_answers(client):
    state=client.get('/api/state').json()
    assert state['document_count']==8
    reply=client.post('/api/answer',json={'question':'Why use an idempotency key for orders?'}).json()
    assert reply['grounded']
    assert reply['citations']
    assert '[1]' in reply['answer']
    assert 'idempotency' in ' '.join(x['title'].lower() for x in reply['citations'])

def test_filter_collection(client):
    response=client.post('/api/search',json={'question':'What is JWT replay prevention?','collection':'security'}).json()
    assert response['matches']
    assert all(x['collection']=='security' for x in response['matches'])

def test_ingestion_and_update(client):
    document={'title':'Custom Load Shedding','collection':'engineering','content':('Load shedding protects overloaded web servers by rejecting excess requests. ' * 5)}
    r=client.post('/api/documents',json=document)
    assert r.status_code==201
    first=r.json()['document_id']
    assert 'Custom Load Shedding' in [x['title'] for x in client.post('/api/search',json={'question':'load shedding'}).json()['matches']]
    document['content']='Backpressure is a control mechanism. '*8
    assert client.post('/api/documents',json=document).json()['document_id']==first
    assert client.get('/api/state').json()['document_count']==9

def test_evaluation_and_validation(client):
    score=client.get('/api/evaluation').json()
    assert score['queries']==6
    assert 0<=score['recall_at_5']<=1
    assert 0<=score['mrr_at_5']<=1
    assert client.post('/api/answer',json={'question':'bad'}).status_code==422

def test_insufficient_evidence(client):
    reply=client.post('/api/answer',json={'question':'qzxzv nonsense random'}).json()
    assert reply['grounded'] is False

def test_reindex_removes_stale_chunk_terms(client):
    doc={'title':'Temporary Guide','collection':'ai','content':'turbogadget stability analysis and segmentation. '*8}
    client.post('/api/documents',json=doc)
    assert any(x['title']=='Temporary Guide' for x in client.post('/api/search',json={'question':'turbogadget'}).json()['matches'])
    doc['content']='bloomfilters document deduplication and search index. '*8
    client.post('/api/documents',json=doc)
    assert not any(x['title']=='Temporary Guide' for x in client.post('/api/search',json={'question':'turbogadget'}).json()['matches'])
