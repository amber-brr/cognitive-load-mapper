# Cognitive Load Mapper

Amber Jiang

https://cogload-frontend-98299138336.us-west2.run.app

## Overview

Analyzes how cognitive load flows through long-form articles. Scrapes Substack publications, extracts per-paragraph complexity features, clusters articles into gradient shapes (plateau, rollercoaster, resolution), and serves an interactive Streamlit app that visualizes complexity trajectories and rewrites flagged paragraphs toward a target shape using an LLM.

## Repo Structure

```
cognitive-load-mapper/
├── app.py                     # Streamlit frontend
├── api/
│   ├── main.py                # FastAPI app (endpoints)
│   ├── data.py                # Data loading and querying
│   ├── rewrite.py             # LLM rewrite logic
│   └── schemas.py             # Pydantic models
├── scripts/
│   ├── data_collector.py      # Scrape Substack articles
│   ├── para_features.py       # Per-paragraph feature extraction
│   ├── article_features.py    # Article-level feature aggregation
│   ├── cluster_shapes.py      # K-means shape labeling
│   ├── compare_complexity.py  # Complexity metric comparison
│   ├── train.py               # Engagement prediction models
│   ├── eda.py                 # Exploratory analysis
│   └── fix_titles.py          # Title cleanup utility
├── tests/
├── processed/                 # Processed CSVs (pipeline output)
├── outputs/                   # Model results
├── Dockerfile                 # Streamlit app container
├── docker-compose.yml         # Runs API + app together
├── requirements.txt
└── .env.example
```

## Setup

```bash
# Install dependencies
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# NLP model data (required once)
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('vader_lexicon')"

# Copy and fill in env vars
cp .env.example .env
```

**`.env` variables:**

```
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free
DATA_DIR=data/processed
```

## Pipeline

Run scripts in order to build the dataset from scratch:

**1. Collect articles**

```bash
python scripts/data_collector.py --output-dir data/processed --max-posts-per-publication 200
```

**2. Extract paragraph features**

```bash
python scripts/para_features.py
```

**3. Aggregate article features**

```bash
python scripts/article_features.py
```

**4. Label gradient shapes (k-means)**

```bash
python scripts/cluster_shapes.py
```

**5. Train engagement prediction models**

```bash
python scripts/train.py
```

## Running the App

### With Docker Compose (recommended)

```bash
docker compose up --build
```

- API: `http://localhost:8080`
- App: `http://localhost:8501`

### Locally

```bash
# Terminal 1 — API
uvicorn api.main:app --host 0.0.0.0 --port 8080

# Terminal 2 — Streamlit
streamlit run app.py
```

## API Endpoints

| Method | Path                     | Description                                      |
| ------ | ------------------------ | ------------------------------------------------ |
| GET    | `/health`                | Health check                                     |
| GET    | `/ready`                 | Readiness check (data loaded)                    |
| GET    | `/articles`              | List articles; filter by `publication`, `shape`  |
| GET    | `/articles/{id}`         | Article detail with paragraphs                   |
| POST   | `/articles/{id}/rewrite` | Rewrite flagged paragraphs toward a target shape |

Interactive docs at `http://localhost:8080/docs`.

## Tests

```bash
pytest tests/ -v
```

## Gradient Shapes

| Shape         | Description                         |
| ------------- | ----------------------------------- |
| plateau       | Consistent complexity throughout    |
| rollercoaster | Alternating high and low complexity |
| resolution    | Complexity fades toward the end     |
| ramp          | Complexity builds toward the end    |
| cliff         | Dense opening, simpler after        |
