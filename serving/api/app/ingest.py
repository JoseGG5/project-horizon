from __future__ import annotations

import argparse
import os

import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy.dialects.postgresql import insert

from app.database import engine
from app.models import metadata, projects_table


DATABASE_URL = os.getenv("DATABASE_URL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest project data and embeddings into PostgreSQL + pgvector.",
    )
    parser.add_argument(
        "--projects-csv",
        default="/data/fp7_projects/project.csv",
        help="Path to the source project CSV inside the container.",
    )
    parser.add_argument(
        "--model-path",
        default="/models/checkpoint-600",
        help="Local SentenceTransformer model path inside the container.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding and insert batch size.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of projects to ingest.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Delete existing rows before ingesting.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str | None:
    """Normalizes a CSV text field before building a project record.

    Args:
        value (object): Raw value read from the CSV.

    Returns:
        str | None: The stripped text value, or ``None`` if the value is
        missing or empty after trimming whitespace.
    """
    if pd.isna(value):
        return None
    text_value = str(value).strip()
    return text_value or None


def load_projects(path: str, limit: int | None) -> list[dict[str, object]]:
    """Loads projects to insert on db

    Args:
        path (str): Path to csv file
        limit (int | None): Number of projects to load

    Returns:
        list[dict[str, object]]: The list of projects
    """
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        quotechar='"',
        usecols=["id", "acronym", "title", "frameworkProgramme", "objective"],
        dtype=str,
    )

    if limit is not None:
        df = df.head(limit)

    records: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        records.append(
            {
                "id": int(getattr(row, "id")),
                "acronym": clean_text(getattr(row, "acronym")),
                "title": clean_text(getattr(row, "title")) or "",
                "framework_programme": clean_text(getattr(row, "frameworkProgramme")),
                "objective": clean_text(getattr(row, "objective")),
            }
        )
    return records


def ensure_schema() -> None:
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        metadata.create_all(connection)


def ingest_projects(
    records: list[dict[str, object]],
    model_path: str,
    batch_size: int,
    truncate: bool,
) -> None:
    """Ingests projects in database.

    Args:
        records (list[dict[str, object]]): A list of projects in dictionary format
        model_path (str): Path to weights artifact
        batch_size (int): The batch size to use for inserts and embeddings
        truncate (bool): In case table needs to be deleted
    """
    model = SentenceTransformer(model_path)

    with engine.connect() as connection:
        if truncate:
            connection.execute(projects_table.delete())
            connection.commit()

        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts = [(record["objective"] or "") for record in batch]
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            rows = []
            for record, embedding in zip(batch, embeddings.tolist(), strict=True):
                rows.append(
                    {
                        **record,
                        "embedding": embedding,
                    }
                )

            stmt = insert(projects_table).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=[projects_table.c.id],
                set_={
                    "acronym": stmt.excluded.acronym,
                    "title": stmt.excluded.title,
                    "framework_programme": stmt.excluded.framework_programme,
                    "objective": stmt.excluded.objective,
                    "embedding": stmt.excluded.embedding,
                },
            )
            connection.execute(stmt)
            connection.commit()

            print(f"Ingested {min(start + batch_size, len(records))}/{len(records)} projects")


def main() -> None:
    args = parse_args()

    print(f"Connecting to {DATABASE_URL.rsplit('@', 1)[-1]}")
    ensure_schema()

    records = load_projects(args.projects_csv, args.limit)
    print(f"Loaded {len(records)} projects from {args.projects_csv}")

    ingest_projects(
        records=records,
        model_path=args.model_path,
        batch_size=args.batch_size,
        truncate=args.truncate,
    )

    print("Ingestion completed.")


if __name__ == "__main__":
    main()
