import strawberry
from strawberry.types import Info

from app import models
from app.graphql.types import Tool, tool_from_model


@strawberry.input
class CreateToolInput:
    name: str
    description: str
    version: str
    provider_id: int
    category_id: int


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_tool(self, info: Info, input: CreateToolInput) -> Tool:
        async with info.context["session_factory"]() as session:
            tool = models.Tool(**strawberry.asdict(input))
            session.add(tool)
            await session.commit()
            await session.refresh(tool)
            return tool_from_model(tool)

    @strawberry.mutation
    async def delete_tool(self, info: Info, id: int) -> bool:
        async with info.context["session_factory"]() as session:
            tool = await session.get(models.Tool, id)
            if not tool:
                return False
            await session.delete(tool)
            await session.commit()
            return True
