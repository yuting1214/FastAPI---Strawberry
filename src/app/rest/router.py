"""
REST API Routes - Two Approaches:

1. Traditional REST: Separate endpoints (client makes 6 requests)
2. Aggregated REST: Single endpoint with joins (server does the work)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_session
from app.models import Tool, Provider, Category, Parameter, Example
from app.rest.schemas import (
    ToolResponse,
    ToolListItem,
    ProviderResponse,
    CategoryResponse,
    ParameterResponse,
    ExampleResponse,
    ToolDetailResponse,
)

router = APIRouter(prefix="/api/v1", tags=["REST API"])


# =============================================================================
# Approach 1: Traditional REST - Separate Endpoints
# Client must make 6 requests to build Tool Detail Page
# =============================================================================

@router.get("/tools", response_model=list[ToolResponse])
async def list_tools(
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    """List all tools."""
    stmt = select(Tool).limit(limit).offset(offset)
    result = await session.execute(stmt)
    return result.scalars().all()


@router.get("/tools/{tool_id}", response_model=ToolResponse)
async def get_tool(tool_id: int, session: AsyncSession = Depends(get_session)):
    """Get tool basic info - Request 1 of 6."""
    tool = await session.get(Tool, tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool


@router.get("/tools/{tool_id}/parameters", response_model=list[ParameterResponse])
async def get_tool_parameters(tool_id: int, session: AsyncSession = Depends(get_session)):
    """Get tool parameters - Request 2 of 6."""
    stmt = select(Parameter).where(Parameter.tool_id == tool_id)
    result = await session.execute(stmt)
    return result.scalars().all()


@router.get("/tools/{tool_id}/examples", response_model=list[ExampleResponse])
async def get_tool_examples(tool_id: int, session: AsyncSession = Depends(get_session)):
    """Get tool examples - Request 3 of 6."""
    stmt = select(Example).where(Example.tool_id == tool_id)
    result = await session.execute(stmt)
    return result.scalars().all()


@router.get("/providers/{provider_id}", response_model=ProviderResponse)
async def get_provider(provider_id: int, session: AsyncSession = Depends(get_session)):
    """Get provider info - Request 4 of 6."""
    provider = await session.get(Provider, provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    return provider


@router.get("/categories/{category_id}", response_model=CategoryResponse)
async def get_category(category_id: int, session: AsyncSession = Depends(get_session)):
    """Get category info - Request 5 of 6."""
    category = await session.get(Category, category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    return category


@router.get("/categories/{category_id}/tools", response_model=list[ToolListItem])
async def get_tools_by_category(
    category_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get tools in category (for related tools) - Request 6 of 6."""
    stmt = select(Tool).where(Tool.category_id == category_id)
    result = await session.execute(stmt)
    return result.scalars().all()


# =============================================================================
# Approach 2: Aggregated REST - Single Endpoint with Eager Loading
# Server does the joining, client makes 1 request
# =============================================================================

@router.get("/tools/{tool_id}/detail", response_model=ToolDetailResponse)
async def get_tool_detail(tool_id: int, session: AsyncSession = Depends(get_session)):
    """
    Get complete tool detail in ONE request.
    Uses eager loading (selectinload) to fetch all relations.
    This is the "optimized REST" approach.
    """
    stmt = (
        select(Tool)
        .where(Tool.id == tool_id)
        .options(
            selectinload(Tool.provider),
            selectinload(Tool.category),
            selectinload(Tool.parameters),
            selectinload(Tool.examples),
        )
    )
    result = await session.execute(stmt)
    tool = result.scalar_one_or_none()

    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")

    # Fetch related tools (same category, excluding self)
    related_stmt = (
        select(Tool)
        .where(Tool.category_id == tool.category_id)
        .where(Tool.id != tool.id)
        .limit(5)
    )
    related_result = await session.execute(related_stmt)
    related_tools = related_result.scalars().all()

    return ToolDetailResponse(
        id=tool.id,
        name=tool.name,
        description=tool.description,
        version=tool.version,
        is_active=tool.is_active,
        usage_count=tool.usage_count,
        created_at=tool.created_at,
        provider=tool.provider,
        category=tool.category,
        parameters=tool.parameters,
        examples=tool.examples,
        related_tools=related_tools,
    )
