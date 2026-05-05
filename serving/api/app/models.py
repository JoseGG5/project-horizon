import os

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Column, MetaData, Table, Text


EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))

metadata = MetaData()

projects_table = Table(
    "projects",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("acronym", Text, nullable=True),
    Column("title", Text, nullable=False),
    Column("framework_programme", Text, nullable=True),
    Column("objective", Text, nullable=True),
    Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=False),
)
