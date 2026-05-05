import os

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import metadata, projects_table
from app.ray_client import EmbeddingClient, RAY_SERVE_URL
from app.schemas import ProjectCreate, ProjectRead, SearchRequest, SearchResult


app = FastAPI(title="Projects API")

DATABASE_URL = os.getenv("DATABASE_URL")
embedding_client = EmbeddingClient(RAY_SERVE_URL)


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, object]:
    """Health endpoint to monitor service health

    Args:
        db (Session, optional): A db dependency injection to ensure that connection
        with db is ok.

    Returns:
        dict[str, object]: Dictionary containing status.
    """
    db.execute(text("SELECT 1"))  # just to ensure db connection is fine
    return {
        "status": "ok",
        "database": "reachable"
    }


@app.post("/projects", response_model=ProjectRead, status_code=201)
async def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
) -> ProjectRead:
    """Creates a new project and stores its embedding in the database.

    Args:
        project (ProjectCreate): Project payload received from the API request.
        db (Session, optional): Database session provided by FastAPI dependency
        injection.

    Raises:
        HTTPException: Raised when a project with the same id already exists.

    Returns:
        ProjectRead: The created project data returned to the client.
    """
    metadata.create_all(bind=db.bind)

    # check that project is currently not in db
    existing = db.execute(
        select(projects_table.c.id).where(projects_table.c.id == project.id)
    ).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Project already exists")

    # embed the project objective and store in db
    embedding = (await embedding_client.embed([project.objective or ""]))[0]
    db.execute(
        projects_table.insert().values(
            id=project.id,
            acronym=project.acronym,
            title=project.title,
            framework_programme=project.framework_programme,
            objective=project.objective,
            embedding=embedding,
        )
    )
    db.commit()
    return ProjectRead(**project.model_dump())


@app.post("/projects/search", response_model=list[SearchResult])
async def search_projects(
    request: SearchRequest,
    db: Session = Depends(get_db),
) -> list[SearchResult]:
    """Searches projects by semantic similarity against the stored embeddings.

    Args:
        request (SearchRequest): Search text and maximum number of results.
        db (Session, optional): Database session provided by FastAPI dependency
        injection.

    Returns:
        list[SearchResult]: The top matching projects ordered by similarity
        score.
    """
    query_embedding = (await embedding_client.embed([request.query]))[0]

    stmt = text(
        """
        SELECT
            id,
            acronym,
            title,
            framework_programme,
            objective,
            1 - (embedding <=> CAST(:embedding AS vector)) AS score
        FROM projects
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
        """
    )

    rows = db.execute(
        stmt,
        {
            "embedding": str(query_embedding),
            "top_k": request.top_k,
        },
    ).mappings()  # to store in memory as dicts and be able to unwrap

    return [SearchResult(**row) for row in rows]
