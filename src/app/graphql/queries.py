import strawberry
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from strawberry.types import Info

from app import models
from app.graphql.types import Tool, Provider, Category, tool_from_model


@strawberry.type
class Query:
    @strawberry.field
    async def tool(self, info: Info, id: int) -> Tool | None:
        async with info.context["session_factory"]() as session:
            stmt = (
                select(models.Tool)
                .where(models.Tool.id == id)
                .options(
                    selectinload(models.Tool.provider),
                    selectinload(models.Tool.category),
                    selectinload(models.Tool.parameters),
                    selectinload(models.Tool.examples),
                )
            )
            result = await session.execute(stmt)
            tool = result.scalar_one_or_none()
            if tool is None:
                return None
            return tool_from_model(tool)

    @strawberry.field
    async def tools(self, info: Info, limit: int = 20, offset: int = 0) -> list[Tool]:
        async with info.context["session_factory"]() as session:
            stmt = (
                select(models.Tool)
                .options(
                    selectinload(models.Tool.provider),
                    selectinload(models.Tool.category),
                    selectinload(models.Tool.parameters),
                    selectinload(models.Tool.examples),
                )
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            return [tool_from_model(t) for t in result.scalars().all()]

    @strawberry.field
    async def providers(self, info: Info) -> list[Provider]:
        async with info.context["session_factory"]() as session:
            result = await session.execute(select(models.Provider))
            return [
                Provider(
                    id=p.id,
                    name=p.name,
                    logo_url=p.logo_url,
                    website=p.website,
                    created_at=p.created_at,
                )
                for p in result.scalars().all()
            ]

    @strawberry.field
    async def categories(self, info: Info) -> list[Category]:
        async with info.context["session_factory"]() as session:
            result = await session.execute(select(models.Category))
            return [
                Category(
                    id=c.id,
                    name=c.name,
                    description=c.description,
                )
                for c in result.scalars().all()
            ]
