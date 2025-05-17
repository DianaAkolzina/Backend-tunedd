import os
import uuid
import time
import logging
from typing import Dict, List

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.components.embedders.ollama import (
    OllamaDocumentEmbedder, OllamaTextEmbedder
)
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever
from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore

# Setup
load_dotenv("../.env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tunedd_api.main1")

# Load ENV
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL")
RAG_CHAT_MODEL = os.getenv("RAG_CHAT_MODEL")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "http://localhost:8080")

# Validate ENV
for var in [RAG_EMBEDDING_MODEL, RAG_CHAT_MODEL, HUGGING_FACE_API_KEY]:
    if not var:
        raise EnvironmentError("Missing required environment variables.")

app = FastAPI()
templates = Jinja2Templates(directory="tunedd_api/templates")
document_store = WeaviateDocumentStore(url=WEAVIATE_HOST)
conversations: Dict[str, List[Dict[str, str]]] = {}
conversation_docs: Dict[str, List[str]] = {}

prompt_template = """
Based on the documents and chat history, answer the question.

Chat history:
{{ conversation }}

Documents:
{% for document in documents %}
[{{ loop.index }}]: {{ document.metadata.source if document.metadata and document.metadata.source else "No source" }}
---
{{ document.content }}
---
{% endfor %}

Question: {{ question }}
"""

prompt_builder = PromptBuilder(template=prompt_template)
_pipeline_process_query = None

def wait_for_ollama():
    logger.info("Waiting for Ollama to be ready...")
    while True:
        try:
            r = httpx.get(f"{OLLAMA_HOST}/tags")
            if r.status_code == 200:
                logger.info("Ollama is ready.")
                break
        except Exception:
            pass
        time.sleep(1)

def load_and_index_documents():
    logger.info("Loading and indexing documents...")
    wait_for_ollama()

    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter())
    pipeline.add_component("embedder", OllamaDocumentEmbedder(model=RAG_EMBEDDING_MODEL, url=OLLAMA_HOST))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder.documents")
    pipeline.connect("embedder", "writer")

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/ai-agents-arxiv-papers"))
    pdfs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]
    logger.info(f"Found {len(pdfs)} PDF files")

    pipeline.run({"converter": {"sources": pdfs}})

def get_query_pipeline():
    global _pipeline_process_query
    if _pipeline_process_query is None:
        logger.info("Creating query pipeline...")
        _pipeline_process_query = Pipeline()
        _pipeline_process_query.add_component("embedder", OllamaTextEmbedder(model=RAG_EMBEDDING_MODEL, url=OLLAMA_HOST))
        _pipeline_process_query.add_component("retriever", WeaviateEmbeddingRetriever(document_store=document_store))
        _pipeline_process_query.add_component("prompt_builder", prompt_builder)
        _pipeline_process_query.add_component(
            "generator",
            HuggingFaceAPIGenerator(
                api_type="serverless_inference_api",
                api_params={"model": RAG_CHAT_MODEL},
                token=Secret.from_token(HUGGING_FACE_API_KEY),
            )
        )

        _pipeline_process_query.connect("embedder.embedding", "retriever.query_embedding")
        _pipeline_process_query.connect("retriever", "prompt_builder")
        _pipeline_process_query.connect("prompt_builder", "generator")
    return _pipeline_process_query

@app.post("/load")
def load_documents():
    logger.info("Manual trigger to load and index documents.")
    load_and_index_documents()
    return {"status": "Documents loaded"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/ai-agents-arxiv-papers"))
    pdfs = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "conversations": list(conversations.keys()),
        "available_documents": pdfs
})


@app.post("/conversations/create", response_class=HTMLResponse)
def create_conversation(request: Request, documents: str = Form(...)):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/ai-agents-arxiv-papers"))
    all_pdfs = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    doc_list = documents.split(",") if documents else all_pdfs

    cid = str(uuid.uuid4())
    conversations[cid] = []
    conversation_docs[cid] = doc_list
    logger.info(f"Created conversation: {cid} with documents: {doc_list}")
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "conversation_id": cid,
        "messages": []
    })

@app.get("/conversations/{conversation_id}/chat", response_class=HTMLResponse)
def enter_conversation(conversation_id: str, request: Request):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id]
    })

@app.post("/conversations/{conversation_id}/message", response_class=HTMLResponse)
def send_message(request: Request, conversation_id: str, message: str = Form(...)):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    chat_history = conversations[conversation_id]
    pipeline = get_query_pipeline()

    result = pipeline.run({
        "embedder": {"text": message},
        "prompt_builder": {
            "conversation": chat_history,
            "question": message
        }
    })

    reply = result["generator"]["replies"][0]
    chat_history.append({"user": message, "assistant": reply})

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "conversation_id": conversation_id,
        "messages": chat_history
    })

if __name__ == "__main__":
    uvicorn.run("tunedd_api.main1:app", host="0.0.0.0", port=8000, reload=True)