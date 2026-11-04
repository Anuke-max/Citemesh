import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal
from . import core

class Document(BaseModel):
    title: str = Field(min_length=3,max_length=120)
    collection: Literal['engineering','ai','security']
    content: str = Field(min_length=80,max_length=12000)

class Query(BaseModel):
    question: str = Field(min_length=4,max_length=500)
    collection: Literal['all','engineering','ai','security'] = 'all'
    top_k: int = Field(default=5,ge=1,le=10)

@asynccontextmanager
async def lifespan(app):
    core.initialize()
    yield

app=FastAPI(title='CiteMesh Retrieval API',version='1.0.0',lifespan=lifespan)
STATIC=Path(__file__).resolve().parent.parent/'static'/'index.html'

@app.get('/')
def home():return FileResponse(STATIC)

@app.get('/health')
def health():return {'status':'ok','service':'citemesh'}

@app.get('/api/state')
def state():return core.state()

@app.post('/api/documents',status_code=201)
def add_document(payload: Document):
    try:return {'document_id':core.ingest(payload.title,payload.collection,payload.content),'status':'indexed'}
    except ValueError as ex:raise HTTPException(status_code=400,detail=str(ex))

@app.post('/api/search')
def search(query: Query):
    return {'matches':core.search(query.question,query.collection,query.top_k)}

@app.post('/api/answer')
def answer(query: Query):
    return core.answer_question(query.question,query.collection,query.top_k)

@app.get('/api/evaluation')
def evaluate():return core.evaluate()
