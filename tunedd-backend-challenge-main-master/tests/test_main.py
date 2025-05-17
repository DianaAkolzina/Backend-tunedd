import os
import pytest
from fastapi.testclient import TestClient
from tunedd_api.main1 import app

client = TestClient(app)

conversation_id = None
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL")

@pytest.mark.order(1)
def test_manual_document_loading():
    print(">>> Starting document load test")
    response = client.post("/load")
    print(">>> Document load response received")
    assert response.status_code == 200
    assert response.json() == {"status": "Documents loaded"}

@pytest.mark.order(2)
def test_create_conversation_with_docs():
    global conversation_id
    response = client.post("/conversations/create", data={"documents": "doc1.pdf, doc2.pdf"})
    assert response.status_code == 200
    assert "chat.html" in response.text or "Conversation" in response.text
    assert "conversation_id" in response.text or "Send a message" in response.text

    # Extracting conversation_id manually
    # (This assumes it's embedded in the HTML template)
    if "value=\"" in response.text:
        marker = response.text.split("value=\"")[1]
        conversation_id = marker.split("\"")[0]
    assert conversation_id is not None

@pytest.mark.order(3)
def test_enter_existing_conversation():
    response = client.get(f"/conversations/{conversation_id}/chat")
    assert response.status_code == 200
    assert conversation_id in response.text
    assert "Send a message" in response.text

@pytest.mark.order(4)
def test_conversation_pipeline_flow():
    message = "Explain AI agents in simple terms."
    response = client.post(f"/conversations/{conversation_id}/message", data={"message": message})
    assert response.status_code == 200
    assert "Explain AI agents in simple terms." in response.text
    assert "Assistant" in response.text or "Answer" in response.text

@pytest.mark.order(5)
def test_invalid_conversation_chat():
    response = client.get("/conversations/non-existent-id/chat")
    assert response.status_code == 404
    assert "Conversation not found" in response.text or "404" in response.text

@pytest.mark.order(6)
def test_invalid_message_post():
    response = client.post("/conversations/non-existent-id/message", data={"message": "Test?"})
    assert response.status_code == 404
    assert "Conversation not found" in response.text or "404" in response.text
