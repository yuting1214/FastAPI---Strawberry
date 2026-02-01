import pytest


@pytest.mark.anyio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.anyio
async def test_graphql_tools(client):
    query = """
        query {
            tools {
                id
                name
                version
            }
        }
    """
    response = await client.post("/graphql", json={"query": query})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "tools" in data["data"]
    assert len(data["data"]["tools"]) >= 1


@pytest.mark.anyio
async def test_graphql_tool_detail(client):
    """Test fetching a single tool with nested relations."""
    query = """
        query {
            tool(id: 1) {
                id
                name
                version
                usageCount
                provider {
                    name
                    website
                }
                category {
                    name
                }
                parameters {
                    name
                    paramType
                    required
                }
                examples {
                    title
                    inputText
                    outputText
                }
            }
        }
    """
    response = await client.post("/graphql", json={"query": query})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    tool = data["data"]["tool"]
    assert tool is not None
    # Verify structure exists (not specific values since data is random)
    assert "name" in tool
    assert "version" in tool
    assert "usageCount" in tool
    assert tool["provider"] is not None
    assert "name" in tool["provider"]
    assert tool["category"] is not None
    assert "name" in tool["category"]
    assert isinstance(tool["parameters"], list)
    assert isinstance(tool["examples"], list)


@pytest.mark.anyio
async def test_graphql_providers(client):
    query = """
        query {
            providers {
                id
                name
                website
            }
        }
    """
    response = await client.post("/graphql", json={"query": query})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    providers = data["data"]["providers"]
    assert len(providers) >= 3  # At least some providers
    # Verify structure
    for provider in providers:
        assert "id" in provider
        assert "name" in provider
        assert "website" in provider


@pytest.mark.anyio
async def test_graphql_categories(client):
    query = """
        query {
            categories {
                id
                name
                description
            }
        }
    """
    response = await client.post("/graphql", json={"query": query})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    categories = data["data"]["categories"]
    assert len(categories) >= 4  # At least some categories
    # Verify structure
    for category in categories:
        assert "id" in category
        assert "name" in category
        assert "description" in category


@pytest.mark.anyio
async def test_rest_tools_list(client):
    """Test REST API tools list endpoint."""
    response = await client.get("/api/v1/tools")
    assert response.status_code == 200
    tools = response.json()
    assert len(tools) >= 1


@pytest.mark.anyio
async def test_rest_tool_detail(client):
    """Test REST API aggregated tool detail endpoint."""
    response = await client.get("/api/v1/tools/1/detail")
    assert response.status_code == 200
    data = response.json()
    # Verify structure exists
    assert "name" in data
    assert "version" in data
    assert data["provider"] is not None
    assert "name" in data["provider"]
    assert data["category"] is not None
    assert "name" in data["category"]
    assert isinstance(data["parameters"], list)
    assert isinstance(data["examples"], list)
    assert isinstance(data["related_tools"], list)


@pytest.mark.anyio
async def test_rest_tool_not_found(client):
    """Test 404 response for non-existent tool."""
    response = await client.get("/api/v1/tools/999999")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_graphql_create_and_delete_tool(client):
    """Test GraphQL mutations for creating and deleting a tool."""
    # First get a valid provider and category ID
    query = """
        query {
            providers { id }
            categories { id }
        }
    """
    response = await client.post("/graphql", json={"query": query})
    data = response.json()
    provider_id = data["data"]["providers"][0]["id"]
    category_id = data["data"]["categories"][0]["id"]

    # Create tool
    create_mutation = """
        mutation CreateTool($input: CreateToolInput!) {
            createTool(input: $input) {
                id
                name
                version
            }
        }
    """
    variables = {
        "input": {
            "name": "Test Tool",
            "description": "A test tool",
            "version": "1.0.0",
            "providerId": provider_id,
            "categoryId": category_id
        }
    }
    response = await client.post("/graphql", json={"query": create_mutation, "variables": variables})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    created_tool = data["data"]["createTool"]
    assert created_tool["name"] == "Test Tool"
    tool_id = created_tool["id"]

    # Delete tool
    delete_mutation = """
        mutation DeleteTool($id: Int!) {
            deleteTool(id: $id)
        }
    """
    response = await client.post("/graphql", json={"query": delete_mutation, "variables": {"id": tool_id}})
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["deleteTool"] is True

    # Verify deleted
    response = await client.post("/graphql", json={"query": delete_mutation, "variables": {"id": tool_id}})
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["deleteTool"] is False


@pytest.mark.anyio
async def test_graphql_related_tools(client):
    """Test that related tools are fetched correctly."""
    query = """
        query {
            tool(id: 1) {
                id
                categoryId
                relatedTools {
                    id
                    name
                }
            }
        }
    """
    response = await client.post("/graphql", json={"query": query})
    assert response.status_code == 200
    data = response.json()
    tool = data["data"]["tool"]
    assert tool is not None
    # Related tools should not include the tool itself
    for related in tool["relatedTools"]:
        assert related["id"] != tool["id"]


@pytest.mark.anyio
async def test_rest_pagination(client):
    """Test REST API pagination."""
    response = await client.get("/api/v1/tools?limit=2&offset=0")
    assert response.status_code == 200
    page1 = response.json()
    assert len(page1) <= 2

    response = await client.get("/api/v1/tools?limit=2&offset=2")
    assert response.status_code == 200
    page2 = response.json()

    # Pages should be different (if enough data)
    if len(page1) == 2 and len(page2) > 0:
        assert page1[0]["id"] != page2[0]["id"]
