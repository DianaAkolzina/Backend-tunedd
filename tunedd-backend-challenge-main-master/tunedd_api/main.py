import logging
import os
import uuid

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.components.embedders.ollama import (
    OllamaDocumentEmbedder, OllamaTextEmbedder)
from haystack_integrations.components.retrievers.weaviate import \
    WeaviateEmbeddingRetriever
from haystack_integrations.document_stores.weaviate.document_store import \
    WeaviateDocumentStore

# Initialise FastAPI
app = FastAPI()

# Initialise logger
logger = logging.getLogger("uvicorn.info")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | "
    "%(module)s:%(funcName)s:%(lineno)d - %(message)s",
)

# Load environment variables
load_dotenv("../.env")
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", None)
if RAG_EMBEDDING_MODEL is None:
    raise Exception("RAG_EMBEDDING_MODEL environment variable is not defined")

RAG_CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", None)
if RAG_CHAT_MODEL is None:
    raise Exception("RAG_CHAT_MODEL environment variable is not defined")

HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY", None)
if HUGGING_FACE_API_KEY is None:
    raise Exception("HUGGING_FACE_API_KEY environment variable is not defined")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost/11434")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "http://localhost/8080")

# Create global variable for document store
document_store = WeaviateDocumentStore(url=WEAVIATE_HOST)

# Create database
conversations = {}

# Create prompt template
template = """
        Based on information in the provided documents and the current chat history, answer the question.
        
        Chat history: {{conversation}}

        Documents:
        {% for document in documents %}
        [Document {{loop.index}}]: {{document.metadata.source if document.metadata.source else "Document " + loop.index|string}}
        ---
        {{document.content}}
        ---
        {% endfor %}

        Question: {{question}}           
        """
prompt_builder = PromptBuilder(template=template)

# Create global variable for query processing pipeline so
# it can be instantiated once and used by the chat endpoint multiple times
_pipeline_process_query = None


def process_documents():
    """
    Process PDFs through pipeline
    """
    logger.info("Processing documents")

    # Create pipeline
    _pipeline_process_documents = Pipeline()
    _pipeline_process_documents.add_component("converter", PyPDFToDocument())
    _pipeline_process_documents.add_component("cleaner", DocumentCleaner())
    _pipeline_process_documents.add_component("splitter", DocumentSplitter())
    _pipeline_process_documents.add_component(
        "embedder", OllamaDocumentEmbedder(model=RAG_EMBEDDING_MODEL, url=OLLAMA_HOST)
    )
    _pipeline_process_documents.add_component("writer", DocumentWriter(document_store))

    _pipeline_process_documents.connect("converter", "cleaner")
    _pipeline_process_documents.connect("cleaner", "splitter")
    _pipeline_process_documents.connect("splitter", "embedder.documents")
    _pipeline_process_documents.connect("embedder", "writer")

    # Directory containing the pdfs
    data_directory = "../data/ai-agents-arxiv-papers"

    # List of paths of documents that require processing
    documentsToProcess = []

    # Only get PDF files
    for filename in os.listdir(data_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_directory, filename)
            try:
                # Add to list of documents
                documentsToProcess.append(file_path)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    # Process documents
    logger.info(f"Processing {len(documentsToProcess)} documents")
    _pipeline_process_documents.run({"converter": {"sources": documentsToProcess}})


@app.post("/conversations/")
def create_conversation():
    """
    Creates a new conversation
    Returns - Conversation ID
    """

    # Create ID
    conversation_id = str(uuid.uuid4())

    # Initialise conversation
    conversations[conversation_id] = []

    return {"conversation_id": conversation_id}


@app.post("/conversations/{conversation_id}/message")
def send_message(conversation_id: str, message: str):
    """
    Processes a conversation message, retrieves its context, and generates a response
    """

    # Make sure the conversation exists
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation does not exist")

    # Generate response
    response = get_pipeline().run(
        {
            "embedder": {"text": message},
            "prompt_builder": {
                "conversation": conversations[conversation_id],
                "question": message,
            },
        }
    )

    # Add to chat history
    conversations[conversation_id].append({"user": message, "assistant": response})

    return {"response": response}


def get_pipeline():
    """
    Returns the pipeline for query processing, setting it up if it has not been yet
    """
    global _pipeline_process_query
    if _pipeline_process_query is None:
        # Set up global pipeline variable
        _pipeline_process_query = Pipeline()

        _pipeline_process_query.add_component(
            "embedder", OllamaTextEmbedder(model=RAG_EMBEDDING_MODEL, url=OLLAMA_HOST)
        )
        _pipeline_process_query.add_component(
            "retriever", WeaviateEmbeddingRetriever(document_store=document_store)
        )
        _pipeline_process_query.add_component("prompt_builder", prompt_builder)
        _pipeline_process_query.add_component(
            "generator",
            HuggingFaceAPIGenerator(
                api_type="serverless_inference_api",
                api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
                token=Secret.from_token(HUGGING_FACE_API_KEY),
            ),
        )

        _pipeline_process_query.connect(
            "embedder.embedding", "retriever.query_embedding"
        )
        _pipeline_process_query.connect("retriever", "prompt_builder")
        _pipeline_process_query.connect("prompt_builder", "generator")

    return _pipeline_process_query


def start():
    process_documents()

    # Run server
    uvicorn.run(
        "tunedd_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        log_config=None,
    )
