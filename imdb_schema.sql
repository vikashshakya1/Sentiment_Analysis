CREATE TABLE imdb_reviews (
    id SERIAL PRIMARY KEY,
    review_text TEXT NOT NULL,
    sentiment VARCHAR(10) NOT NULL CHECK (sentiment IN ('positive', 'negative'))
);