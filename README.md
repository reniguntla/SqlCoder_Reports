# SQLCoder Reports (Streamlit + PostgreSQL + Ollama)

A Streamlit app that converts natural language questions to safe SQL using a locally running SQLCoder model via Ollama, executes the query on a local PostgreSQL database in **read-only mode**, and formats results for non-technical users.

## Features

- Natural language to SQL generation using local SQLCoder (`ollama`).
- PostgreSQL schema introspection (tables, columns, data types, PKs, FKs, comments).
- Read-only safety checks (rejects destructive statements and multiple statements).
- SQL execution with graceful error handling.
- Result rendering as table + AI-generated explanation.
- Streamlit UI with optional SQL transparency and query history.

## Prerequisites

1. Python 3.10+
2. Local PostgreSQL instance
3. Ollama installed and running
4. SQLCoder model pulled into Ollama, for example:
   ```bash
   ollama pull sqlcoder
   ```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
PGHOST=localhost
PGPORT=5432
PGUSER=postgres
PGPASSWORD=postgres
PGDATABASE=postgres
PGSCHEMA=public

OLLAMA_HOST=http://localhost:11434
SQLCODER_MODEL=sqlcoder
```

## Run

```bash
streamlit run app.py
```

## Example prompt

"List top 5 employees with highest salary along with their name, employee ID, and salary."

The app will:
1. Read schema metadata.
2. Ask SQLCoder to generate SQL.
3. Validate SQL as read-only.
4. Execute query.
5. Display table + summary explanation.

## Security notes

- Database connection is configured with `default_transaction_read_only=on`.
- SQL is validated to allow only `SELECT` / `WITH` statements.
- Common destructive keywords are blocked.

## Extensibility

- You can switch model by changing `SQLCODER_MODEL`.
- Provider logic is isolated around Ollama client calls in `generate_sql` and `explain_results`, making adapter-based extension straightforward.
