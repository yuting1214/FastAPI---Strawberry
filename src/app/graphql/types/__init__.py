"""
GraphQL Types with manual definitions for proper async support.

We use manual type definitions instead of strawberry-sqlalchemy-mapper's automatic
relationship resolution because the mapper has compatibility issues with async sessions.
The relationships are eagerly loaded using SQLAlchemy's selectinload in the queries.
"""
import strawberry
from datetime import datetime
from sqlalchemy import select

from app import models


@strawberry.type
class Provider:
    id: int
    name: str
    logo_url: str
    website: str
    created_at: datetime


@strawberry.type
class Category:
    id: int
    name: str
    description: str


@strawberry.type
class Parameter:
    id: int
    name: str
    param_type: str
    required: bool
    description: str
    default_value: str | None


@strawberry.type
class Example:
    id: int
    title: str
    input_text: str
    output_text: str


@strawberry.type
class Tool:
    id: int
    name: str
    description: str
    version: str
    is_active: bool
    usage_count: int
    created_at: datetime
    provider_id: int
    category_id: int

    # These will be populated from eagerly-loaded relationships
    _provider: strawberry.Private[models.Provider | None] = None
    _category: strawberry.Private[models.Category | None] = None
    _parameters: strawberry.Private[list[models.Parameter] | None] = None
    _examples: strawberry.Private[list[models.Example] | None] = None

    @strawberry.field
    def provider(self) -> Provider | None:
        if self._provider is None:
            return None
        return Provider(
            id=self._provider.id,
            name=self._provider.name,
            logo_url=self._provider.logo_url,
            website=self._provider.website,
            created_at=self._provider.created_at,
        )

    @strawberry.field
    def category(self) -> Category | None:
        if self._category is None:
            return None
        return Category(
            id=self._category.id,
            name=self._category.name,
            description=self._category.description,
        )

    @strawberry.field
    def parameters(self) -> list[Parameter]:
        if self._parameters is None:
            return []
        return [
            Parameter(
                id=p.id,
                name=p.name,
                param_type=p.param_type,
                required=p.required,
                description=p.description,
                default_value=p.default_value,
            )
            for p in self._parameters
        ]

    @strawberry.field
    def examples(self) -> list[Example]:
        if self._examples is None:
            return []
        return [
            Example(
                id=e.id,
                title=e.title,
                input_text=e.input_text,
                output_text=e.output_text,
            )
            for e in self._examples
        ]

    @strawberry.field
    async def related_tools(self, info) -> list["Tool"]:
        """Custom resolver: Get other tools in the same category."""
        session = info.context["session"]
        stmt = (
            select(models.Tool)
            .where(models.Tool.category_id == self.category_id)
            .where(models.Tool.id != self.id)
            .limit(5)
        )
        result = await session.execute(stmt)
        tools = result.scalars().all()
        return [tool_from_model(t) for t in tools]


def tool_from_model(tool: models.Tool, include_relations: bool = True) -> Tool:
    """Convert SQLAlchemy Tool model to Strawberry Tool type.

    Args:
        tool: The SQLAlchemy Tool model
        include_relations: Whether to include eagerly-loaded relationships.
                          Set to False when relationships aren't loaded.
    """
    from sqlalchemy.orm import object_session
    from sqlalchemy.inspection import inspect

    # Helper to safely get relationship if it's loaded
    def get_if_loaded(attr_name):
        if not include_relations:
            return None
        state = inspect(tool)
        if attr_name in state.unloaded:
            return None
        return getattr(tool, attr_name, None)

    provider = get_if_loaded('provider')
    category = get_if_loaded('category')
    parameters = get_if_loaded('parameters')
    examples = get_if_loaded('examples')

    return Tool(
        id=tool.id,
        name=tool.name,
        description=tool.description,
        version=tool.version,
        is_active=tool.is_active,
        usage_count=tool.usage_count,
        created_at=tool.created_at,
        provider_id=tool.provider_id,
        category_id=tool.category_id,
        _provider=provider,
        _category=category,
        _parameters=list(parameters) if parameters else None,
        _examples=list(examples) if examples else None,
    )


# Placeholder for compatibility - not used with manual types
class _Mapper:
    mapped_types = {}

mapper = _Mapper()
