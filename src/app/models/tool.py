from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, String, Text, Boolean, Integer, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.provider import Provider
    from app.models.category import Category


class Tool(Base):
    __tablename__ = "tool"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text)
    version: Mapped[str] = mapped_column(String(20))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Foreign keys
    provider_id: Mapped[int] = mapped_column(ForeignKey("provider.id"))
    category_id: Mapped[int] = mapped_column(ForeignKey("category.id"))

    # Relationships -> auto-mapped to GraphQL nested types
    provider: Mapped["Provider"] = relationship(back_populates="tools")
    category: Mapped["Category"] = relationship(back_populates="tools")
    parameters: Mapped[list["Parameter"]] = relationship(
        back_populates="tool", cascade="all, delete-orphan"
    )
    examples: Mapped[list["Example"]] = relationship(
        back_populates="tool", cascade="all, delete-orphan"
    )


class Parameter(Base):
    __tablename__ = "parameter"

    id: Mapped[int] = mapped_column(primary_key=True)
    tool_id: Mapped[int] = mapped_column(ForeignKey("tool.id"))
    name: Mapped[str] = mapped_column(String(100))
    param_type: Mapped[str] = mapped_column(String(50))
    required: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[str] = mapped_column(Text)
    default_value: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    tool: Mapped["Tool"] = relationship(back_populates="parameters")


class Example(Base):
    __tablename__ = "example"

    id: Mapped[int] = mapped_column(primary_key=True)
    tool_id: Mapped[int] = mapped_column(ForeignKey("tool.id"))
    title: Mapped[str] = mapped_column(String(200))
    input_text: Mapped[str] = mapped_column(Text)
    output_text: Mapped[str] = mapped_column(Text)

    tool: Mapped["Tool"] = relationship(back_populates="examples")
