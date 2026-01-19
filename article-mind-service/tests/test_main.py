"""Tests for main application."""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


def test_root(client: TestClient) -> None:
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Article Mind Service API"}


@pytest.mark.asyncio
async def test_root_async(async_client: AsyncClient) -> None:
    """Test root endpoint with async client."""
    response = await async_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Article Mind Service API"}


def test_openapi_docs(client: TestClient) -> None:
    """Test OpenAPI documentation is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200
    openapi = response.json()
    assert openapi["info"]["title"] == "Article Mind Service"
