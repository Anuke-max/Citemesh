"""CiteMesh: local hybrid retrieval with extractive, source-cited answers."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import math
import os
import re
import sqlite3
import time
from pathlib import Path

DB_PATH = os.environ.get('CITEMESH_DB', str(Path(__file__).resolve().parent.parent/'citemesh.db'))
WORDS = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,}")
GOLDEN_QUERIES = [
    ('How do idempotency keys prevent duplicate orders?', 'Idempotency and Safe API Retries'),
    ('What happens after delivery retries are exhausted?', 'Dead Letter Queues and Retry Backoff'),
    ('How does a transactional outbox prevent lost events?', 'Transactional Outbox Pattern'),
    ('What does p95 latency mean for an API?', 'Service Level Objectives and Latency'),
    ('How can we evaluate document retrieval quality?', 'Evaluating Retrieval Quality'),
    ('What protects an API from JWT token replay?', 'API Authentication and Token Validation'),
]
SEED = [
('Idempotency and Safe API Retries', 'engineering', """A client can retry a write request when network timeouts make the response uncertain. Without an idempotency key, a retry can create a duplicate order or charge. An idempotency key identifies one logical operation across retries. The server atomically stores the key, request fingerprint, and result so repeated requests receive the original response. If the same key is reused with a different payload, the server should reject it as a conflict. Idempotency is especially important for payment, checkout, and provisioning APIs. Keys should have bounded retention, correct tenancy scope, and a documented conflict policy."""),
('Dead Letter Queues and Retry Backoff', 'engineering', """Transient downstream failures should be retried with exponential backoff and jitter to reduce synchronized retry storms. An event becomes a dead-letter event when it exceeds the maximum attempt count or is classified as nonrecoverable. A dead-letter queue keeps failed messages inspectable while the normal consumer continues processing other work. Operators investigate root cause, repair data or dependencies, and explicitly redrive the message. Redrive must be idempotent because previously completed side effects might exist. Monitor dead-letter depth, age, and retry volume rather than only aggregate error counts."""),
('Transactional Outbox Pattern', 'engineering', """A service that writes an order to a database and separately publishes a broker event can fail between the two steps. The transactional outbox pattern writes the business record and an outbox event within the same database transaction. A publisher or worker later reads the outbox and forwards events to downstream consumers. Publication may happen more than once after a crash, so consumers need deduplication or idempotent handlers. Outbox processing progress, lag, retry age, and dead-letter rates are useful operational measurements. Exactly-once delivery across multiple services is not guaranteed by an outbox alone."""),
('Service Level Objectives and Latency', 'engineering', """A service level indicator measures an aspect of service behavior such as request success ratio or response latency. A service level objective sets an internal reliability target based on an indicator. The p95 latency is the duration below which 95 percent of measured requests fall within a specified window. A latency SLO should define the endpoint population, measurement window, exclusions, and sample volume. Error budgets express the amount of unreliability allowed under an objective. Teams can use an error budget to balance releases against reliability risk."""),
('Evaluating Retrieval Quality', 'ai', """A retrieval system should evaluate whether relevant source passages are returned before judging answer quality. Recall at k measures whether a relevant document appears among the top k results. Mean reciprocal rank rewards placing the first relevant document closer to rank one. Precision at k measures how many returned results are relevant when multiple relevance labels are available. Use a held-out set of representative questions with reviewed relevance labels. Analyze failures such as missing synonyms, long chunks, poor text extraction, and stale indexes. A citation is evidence only when the linked passage actually supports the answer."""),
('Hybrid Search and Reciprocal Rank Fusion', 'ai', """Lexical search is effective for identifiers, exact error messages, and domain-specific terms. TF-IDF vector similarity scores chunks using term frequency weighted by inverse document frequency and cosine similarity. Reciprocal rank fusion combines multiple ranked lists without requiring directly comparable raw search scores. RRF assigns each item a score based on its positions in component rankings. Hybrid search can improve retrieval when queries mix exact terminology and general descriptions, but its quality must be measured on held-out queries. Sparse TF-IDF is not equivalent to dense transformer embeddings."""),
('API Authentication and Token Validation', 'security', """API authorization should validate the JWT signature, issuer, audience, expiration, algorithm allowlist, and granted scopes. Reject unsigned or algorithm-confused tokens. Token replay risk can be reduced with short-lived access tokens, TLS, binding tokens to clients where supported, and revocation or nonce strategies for sensitive operations. Store secrets in a secret manager rather than source control. Rate limiting, audit logging, least privilege, and input validation are complementary controls. A public demonstration without authentication must never be described as a secure multi-tenant production system."""),
('Zero Downtime Deployment and Rollback', 'engineering', """A zero downtime deployment typically relies on health checks, capacity overlap, and a load balancer shifting traffic away from unhealthy instances. Schema migrations should be backward compatible when old and new application versions may run concurrently. Release automation should support quick rollback of application code, but database rollback requires explicit planning. Canary releases expose a small population to a new version, making regressions visible before full rollout. Dashboards should track errors, p95 latency, saturation, and business process failure during each rollout."""),
]

def connect():
    db=sqlite3.connect(DB_PATH,timeout=10, isolation_level=None)
    db.row_factory=sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA busy_timeout=10000')
    db.execute('PRAGMA foreign_keys=ON')
    return db


def tokenize(text):
    return [m.group().lower() for m in WORDS.finditer(text)]


def chunks_of(text: str, max_words=96, overlap=18):
    words=text.split()
    if not words: return []
    step=max_words-overlap
    return [' '.join(words[i:i+max_words]) for i in range(0,len(words),step) if len(words[i:i+max_words])>=4]


def initialize():
    with connect() as db:
        db.executescript('''
          CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY,title TEXT NOT NULL,collection TEXT NOT NULL,content TEXT NOT NULL,created_at REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY,document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE, collection TEXT NOT NULL,body TEXT NOT NULL,ordinal INTEGER NOT NULL);
          CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
          CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(body, chunk_id UNINDEXED, tokenize='porter unicode61');
        ''')
        current=db.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
        if not current:
            for title,collection,content in SEED:
                _ingest(db,title,collection,content)


def _ingest(db,title,collection,content):
    doc_id=hashlib.sha256((collection+'::'+title).encode()).hexdigest()[:16]
    now=time.time()
    old_chunks=[x[0] for x in db.execute('SELECT id FROM chunks WHERE document_id=?',(doc_id,))]
    for chunk_id in old_chunks:
        db.execute('DELETE FROM chunks_fts WHERE chunk_id=?',(chunk_id,))
    db.execute('DELETE FROM chunks WHERE document_id=?',(doc_id,))
    db.execute('INSERT OR REPLACE INTO documents(id,title,collection,content,created_at) VALUES(?,?,?,?,?)',(doc_id,title,collection,content,now))
    for i,chunk in enumerate(chunks_of(content)):
        chunk_id=hashlib.sha256(f'{doc_id}:{i}:{chunk}'.encode()).hexdigest()[:20]
        db.execute('INSERT INTO chunks(id,document_id,collection,body,ordinal) VALUES(?,?,?,?,?)',(chunk_id,doc_id,collection,chunk,i))
        db.execute('INSERT INTO chunks_fts(body,chunk_id) VALUES(?,?)',(chunk,chunk_id))
    return doc_id


def ingest(title,collection,content):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        try:
            if db.execute('SELECT COUNT(*) FROM documents').fetchone()[0]>=200:
                doc_id=hashlib.sha256((collection+'::'+title).encode()).hexdigest()[:16]
                if not db.execute('SELECT 1 FROM documents WHERE id=?',(doc_id,)).fetchone():
                    raise ValueError('Demo document limit reached (200)')
            doc_id=_ingest(db,title,collection,content)
            db.execute('COMMIT')
            return doc_id
        except Exception:
            db.execute('ROLLBACK')
            raise


def tfidf_rank(query, rows):
    terms=tokenize(query)
    if not terms:return []
    corpus=[Counter(tokenize(r['body'])) for r in rows]
    q=Counter(terms)
    df=Counter()
    for doc in corpus:df.update(set(doc))
    n=len(corpus)
    def w(t):return math.log((1+n)/(1+df[t]))+1.0
    qnorm=math.sqrt(sum((count*w(t))**2 for t,count in q.items())) or 1
    scores=[]
    for row,d in zip(rows,corpus):
        dot=sum(q[t]*d.get(t,0)*w(t)**2 for t in q)
        norm=math.sqrt(sum((count*w(t))**2 for t,count in d.items())) or 1
        if dot:scores.append((row['id'],dot/(qnorm*norm)))
    return sorted(scores,key=lambda x:-x[1])


def search(query: str, collection='all', k=5):
    terms=tokenize(query)[:16]
    if not terms:return []
    with connect() as db:
        filt='' if collection=='all' else 'WHERE collection=?'
        all_rows=[dict(x) for x in db.execute('SELECT id,document_id,collection,body,ordinal FROM chunks '+filt, () if collection=='all' else (collection,))]
        if not all_rows:return []
        fts=' OR '.join('"'+t.replace('"','')+'"' for t in terms)
        filt2='' if collection=='all' else 'AND c.collection=?'
        fts_rows=db.execute('SELECT c.id AS id FROM chunks_fts f JOIN chunks c ON c.id=f.chunk_id WHERE chunks_fts MATCH ? '+filt2+' ORDER BY bm25(chunks_fts) LIMIT 100',(fts,) if collection=='all' else (fts,collection)).fetchall()
        lexical_ids=[x['id'] for x in fts_rows]
        semantic_ids=[x[0] for x in tfidf_rank(query,all_rows)[:100]]
        rrf=defaultdict(float)
        for ranking in (lexical_ids,semantic_ids):
            for pos,cid in enumerate(ranking,1):rrf[cid]+=1.0/(60+pos)
        byid={r['id']:r for r in all_rows}
        selected=sorted(rrf,key=lambda cid:(-rrf[cid],cid))[:max(1,min(k,20))]
        docids=list({byid[cid]['document_id'] for cid in selected})
        names={r['id']:r['title'] for r in db.execute('SELECT id,title FROM documents WHERE id IN ('+','.join('?' for _ in docids)+')',docids)} if docids else {}
        return [{**byid[cid],'title':names[byid[cid]['document_id']], 'score':round(rrf[cid],5)} for cid in selected]


def answer_question(query,collection='all',k=5):
    matches=search(query,collection,k)
    if not matches:return {'answer':'Insufficient evidence in the indexed documents. Try a different query or collection.','citations':[],'grounded':False,'matches':[]}
    terms=set(tokenize(query))
    citations=[]
    snippets=[]
    used_docs=set()
    for match in matches:
        if match['document_id'] in used_docs:continue
        sentences=re.split(r'(?<=[.!?])\s+',match['body'])
        ranked=sorted(sentences,key=lambda x:sum(t in tokenize(x) for t in terms),reverse=True)
        best=ranked[0] if ranked else match['body']
        overlap=len(set(tokenize(best)) & terms)
        if overlap<1:continue
        idx=len(citations)+1
        snippets.append(best.strip()+f' [{idx}]')
        citations.append({'index':idx,'document_id':match['document_id'],'title':match['title'],'collection':match['collection'],'excerpt':match['body'],'chunk_id':match['id']})
        used_docs.add(match['document_id'])
        if len(citations)==2:break
    if not snippets:return {'answer':'Insufficient evidence in the indexed documents.','citations':[],'grounded':False,'matches':matches}
    return {'answer':' '.join(snippets),'citations':citations,'grounded':True,'matches':matches}


def evaluate():
    rows=[]
    for question,title in GOLDEN_QUERIES:
        results=search(question,'all',5)
        title_order=[]
        for r in results:
            if r['title'] not in title_order:title_order.append(r['title'])
        rank=title_order.index(title)+1 if title in title_order else None
        rows.append({'query':question,'expected_source':title,'rank':rank,'found_at_5':rank is not None})
    n=len(rows)
    return {'queries':n,'recall_at_5':round(sum(x['found_at_5'] for x in rows)/n,3),
            'mrr_at_5':round(sum((1/x['rank'] if x['rank'] else 0) for x in rows)/n,3),'results':rows,
            'note':'Illustrative six-question evaluation on built-in seed documents; not an independent production benchmark.'}


def state():
    with connect() as db:
        docs=[dict(x) for x in db.execute('SELECT id,title,collection,created_at FROM documents ORDER BY created_at DESC LIMIT 200')]
        chunks=db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
    return {'documents':docs,'document_count':len(docs),'chunk_count':chunks,'collections':sorted(set(x['collection'] for x in docs))}
