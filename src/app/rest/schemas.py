"""Pydantic models for REST API responses."""
from datetime import datetime
from pydantic import BaseModel


class ProviderResponse(BaseModel):
    id: int
    name: str
    logo_url: str
    website: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CategoryResponse(BaseModel):
    id: int
    name: str
    description: str

    model_config = {"from_attributes": True}


class ParameterResponse(BaseModel):
    id: int
    name: str
    param_type: str
    required: bool
    description: str
    default_value: str | None = None

    model_config = {"from_attributes": True}


class ExampleResponse(BaseModel):
    id: int
    title: str
    input_text: str
    output_text: str

    model_config = {"from_attributes": True}


class ToolResponse(BaseModel):
    id: int
    name: str
    description: str
    version: str
    is_active: bool
    usage_count: int
    created_at: datetime
    provider_id: int
    category_id: int

    model_config = {"from_attributes": True}


class ToolListItem(BaseModel):
    """Lighter response for list endpoints."""
    id: int
    name: str
    description: str
    version: str

    model_config = {"from_attributes": True}


# Aggregated response for "joined" approach
class ToolDetailResponse(BaseModel):
    """Full tool detail - what GraphQL returns in 1 query."""
    id: int
    name: str
    description: str
    version: str
    is_active: bool
    usage_count: int
    created_at: datetime
    provider: ProviderResponse
    category: CategoryResponse
    parameters: list[ParameterResponse]
    examples: list[ExampleResponse]
    related_tools: list[ToolListItem]

    model_config = {"from_attributes": True}
