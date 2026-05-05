CREATE TABLE IF NOT EXISTS projects (
    id BIGINT PRIMARY KEY,
    acronym TEXT,
    title TEXT NOT NULL,
    framework_programme TEXT,
    objective TEXT,
    embedding VECTOR(768) NOT NULL
);
