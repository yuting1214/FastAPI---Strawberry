import strawberry
from strawberry.fastapi import GraphQLRouter

from app.core.database import AsyncSessionLocal
from app.graphql.queries import Query
from app.graphql.mutations import Mutation


async def get_context():
    """
    Provide session factory to resolvers.

    GraphQL resolvers execute concurrently, but SQLAlchemy async sessions
    don't support concurrent operations. Each resolver gets its own session
    via the factory to avoid conflicts.
    """
    return {
        "session_factory": AsyncSessionLocal,
    }


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
)

graphql_router = GraphQLRouter(
    schema,
    context_getter=get_context,
)
