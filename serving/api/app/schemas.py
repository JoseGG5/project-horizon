from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    id: int
    acronym: str | None = None
    title: str
    framework_programme: str | None = None
    objective: str | None = None


class ProjectRead(BaseModel):
    id: int
    acronym: str | None = None
    title: str
    framework_programme: str | None = None
    objective: str | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


class SearchResult(ProjectRead):
    score: float
