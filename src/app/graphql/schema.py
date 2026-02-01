import strawberry
from strawberry.fastapi import GraphQLRouter

from app.core.database import AsyncSessionLocal
from app.graphql.queries import Query
from app.graphql.mutations import Mutation


async def get_context():
    """
    Provide session to all resolvers.
    We use selectinload for eager loading to prevent N+1 queries.
    """
    async with AsyncSessionLocal() as session:
        yield {
            "session": session,
        }


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
)

graphql_router = GraphQLRouter(
    schema,
    context_getter=get_context,
)
