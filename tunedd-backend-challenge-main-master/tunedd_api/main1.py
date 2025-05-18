import os
import uuid
import time
import logging
from typing import Dict, List

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
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
Using only the information in the documents below and the chat history, answer the question. 
If the answer is not found in the documents, respond with: "I couldn't find relevant information in the indexed documents."

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

pdfs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/ai-agents-arxiv-papers"))
all_pdfs = [f for f in os.listdir(pdfs_dir) if f.endswith(".pdf")]

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

def reset_weaviate_class():
    try:
        client = document_store.client
        logger.warning("Deleting 'Default' class from Weaviate...")
        client.schema.delete_class("Default")
    except Exception as e:
        logger.error(f"Failed to delete class: {e}")
        raise

def ensure_weaviate_schema():
    client = document_store.client
    class_obj = {
        "class": "Default",
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]}
        ]
    }
    try:
        existing = client.schema.get()
        classes = [c["class"] for c in existing["classes"]]
        if "Default" not in classes:
            logger.info("Creating Weaviate class schema with 'source'...")
            client.schema.create_class(class_obj)
        else:
            logger.info("Weaviate class already exists. Verifying 'source' property...")
            current = client.schema.get("Default")
            props = [p["name"] for p in current["properties"]]
            if "source" not in props:
                logger.info("Adding 'source' property to Weaviate schema...")
                client.schema.add_property("Default", {"name": "source", "dataType": ["text"]})
    except Exception as e:
        logger.exception("Error ensuring Weaviate schema")
        raise HTTPException(status_code=500, detail=f"Weaviate schema setup failed: {e}")

def load_and_index_documents():
    logger.info("Loading and indexing documents...")
    wait_for_ollama()
    reset_weaviate_class()
    ensure_weaviate_schema()

    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter())
    pipeline.add_component("embedder", OllamaDocumentEmbedder(model=RAG_EMBEDDING_MODEL, url=OLLAMA_HOST))
    pipeline.add_component("writer", DocumentWriter(
        document_store=document_store,
        policy="upsert",
        meta_fields=["source"]
    ))

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder.documents")
    pipeline.connect("embedder", "writer")

    pdfs = [os.path.join(pdfs_dir, f) for f in all_pdfs]
    logger.info(f"Found {len(pdfs)} PDF files")

    pipeline.run({"converter": {"sources": pdfs}})


@app.post("/load")
def load_documents():
    logger.info("Manual trigger to load and index documents.")
    load_and_index_documents()
    return {"status": "Documents loaded"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "conversations": list(conversations.keys()),
        "documents": all_pdfs
    })

@app.post("/conversations/create", response_class=HTMLResponse)
def create_conversation(request: Request, documents: List[str] = Form(...)):
    doc_list = documents if documents else all_pdfs
    cid = str(uuid.uuid4())
    conversations[cid] = []
    conversation_docs[cid] = doc_list
    logger.info(f"Created conversation: {cid} with documents: {doc_list}")
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "conversation_id": cid,
        "messages": [],
        "documents": doc_list
    })

@app.get("/conversations/{conversation_id}/chat", response_class=HTMLResponse)
def enter_conversation(conversation_id: str, request: Request):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id],
        "documents": conversation_docs.get(conversation_id, [])
    })

@app.post("/conversations/{conversation_id}/message", response_class=HTMLResponse)
def send_message(request: Request, conversation_id: str, message: str = Form(...)):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    chat_history = conversations[conversation_id]
    selected_docs = conversation_docs.get(conversation_id, all_pdfs)

    # Step 1: Get embedding
    embed_pipeline = Pipeline()
    embed_pipeline.add_component("embedder", OllamaTextEmbedder(model=RAG_EMBEDDING_MODEL, url=OLLAMA_HOST))
    result_embed = embed_pipeline.run({"embedder": {"text": message}})
    embedding = result_embed.get("embedder", {}).get("embedding")

    if not embedding:
        logger.error(f"Missing embedding in result: {result_embed}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding.")

    logger.info(f"Embedding OK: {len(embedding)} floats")

    # Step 2: Query Weaviate directly
    try:
        weaviate_client = document_store.client
        collection = weaviate_client.collections.get("Default")

        results = collection.query.near_vector(
            near_vector=embedding,
            limit=5,
            return_properties=["content", "source"]
        )

        documents = []
        for obj in results.objects:
            content = obj.properties.get("content", "")
            source = obj.properties.get("source", "unknown")
            if source in selected_docs:
                documents.append({"content": content, "metadata": {"source": source}})

        logger.info(f"Retrieved {len(documents)} filtered docs from Weaviate")

    except Exception as e:
        logger.exception("Direct Weaviate query failed")
        raise HTTPException(status_code=500, detail=f"Vector search error: {e}")

    if not documents:
        logger.warning("No relevant documents retrieved.")
        chat_history.append({
            "user": message,
            "assistant": "I couldn't find relevant information in the indexed documents to answer your question."
        })
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "conversation_id": conversation_id,
            "messages": chat_history,
            "documents": selected_docs
        })

    # Step 3: Build prompt and generate
    local_prompt_builder = PromptBuilder(template=prompt_template)
    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", local_prompt_builder)
    pipeline.add_component("generator", HuggingFaceAPIGenerator(
        api_type="serverless_inference_api",
        api_params={"model": RAG_CHAT_MODEL},
        token=Secret.from_token(HUGGING_FACE_API_KEY),
    ))
    pipeline.connect("prompt_builder", "generator")

    result = pipeline.run({
        "prompt_builder": {
            "conversation": chat_history,
            "question": message,
            "documents": documents
        }
    })

    reply = result["generator"]["replies"][0]
    chat_history.append({"user": message, "assistant": reply})

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "conversation_id": conversation_id,
        "messages": chat_history,
        "documents": selected_docs
    })

if __name__ == "__main__":
    uvicorn.run("tunedd_api.main1:app", host="0.0.0.0", port=8000, reload=True)
