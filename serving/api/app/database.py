import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """ FastAPI has a powerful dependency injection system.
        This function is used as a dependency for all endpoints that
        need to handle db ops. """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
