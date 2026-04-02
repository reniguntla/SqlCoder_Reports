import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import psycopg
import sqlparse
import streamlit as st
from dotenv import load_dotenv
from ollama import Client

load_dotenv()


@dataclass
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    dbname: str


@dataclass
class ModelConfig:
    host: str
    model: str


def get_db_config() -> DBConfig:
    return DBConfig(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        dbname=os.getenv("PGDATABASE", "postgres"),
    )


def get_model_config() -> ModelConfig:
    return ModelConfig(
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model=os.getenv("SQLCODER_MODEL", "sqlcoder"),
    )


def get_connection(cfg: DBConfig) -> psycopg.Connection:
    return psycopg.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        dbname=cfg.dbname,
        options="-c default_transaction_read_only=on",
        autocommit=True,
    )


def schema_context(conn: psycopg.Connection, schema: str = "public") -> str:
    query = """
    WITH table_data AS (
      SELECT
        t.table_name,
        obj_description((quote_ident(t.table_schema)||'.'||quote_ident(t.table_name))::regclass, 'pg_class') AS table_description
      FROM information_schema.tables t
      WHERE t.table_schema = %(schema)s
        AND t.table_type = 'BASE TABLE'
    ),
    column_data AS (
      SELECT
        c.table_name,
        c.column_name,
        c.data_type,
        c.is_nullable,
        c.column_default,
        col_description((quote_ident(c.table_schema)||'.'||quote_ident(c.table_name))::regclass,
                        c.ordinal_position) AS column_description
      FROM information_schema.columns c
      WHERE c.table_schema = %(schema)s
    ),
    pk_data AS (
      SELECT
        kcu.table_name,
        string_agg(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) AS pk_columns
      FROM information_schema.table_constraints tc
      JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
       AND tc.table_schema = kcu.table_schema
      WHERE tc.table_schema = %(schema)s
        AND tc.constraint_type = 'PRIMARY KEY'
      GROUP BY kcu.table_name
    ),
    fk_data AS (
      SELECT
        tc.table_name,
        string_agg(
          kcu.column_name || ' -> ' || ccu.table_name || '.' || ccu.column_name,
          '; '
        ) AS fk_relations
      FROM information_schema.table_constraints tc
      JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
       AND tc.table_schema = kcu.table_schema
      JOIN information_schema.constraint_column_usage ccu
        ON ccu.constraint_name = tc.constraint_name
       AND ccu.table_schema = tc.table_schema
      WHERE tc.table_schema = %(schema)s
        AND tc.constraint_type = 'FOREIGN KEY'
      GROUP BY tc.table_name
    )
    SELECT
      td.table_name,
      COALESCE(td.table_description, '') AS table_description,
      cd.column_name,
      cd.data_type,
      COALESCE(cd.column_description, '') AS column_description,
      COALESCE(pk.pk_columns, '') AS primary_keys,
      COALESCE(fk.fk_relations, '') AS foreign_keys
    FROM table_data td
    JOIN column_data cd ON td.table_name = cd.table_name
    LEFT JOIN pk_data pk ON td.table_name = pk.table_name
    LEFT JOIN fk_data fk ON td.table_name = fk.table_name
    ORDER BY td.table_name, cd.column_name;
    """

    with conn.cursor() as cur:
        cur.execute(query, {"schema": schema})
        rows = cur.fetchall()

    if not rows:
        return "No schema metadata found."

    grouped: dict[str, list[tuple[Any, ...]]] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)

    lines: list[str] = []
    for table, cols in grouped.items():
        lines.append(f"Table: {table}")
        lines.append(f"Description: {cols[0][1] or 'N/A'}")
        lines.append(f"Primary keys: {cols[0][5] or 'N/A'}")
        lines.append(f"Foreign keys: {cols[0][6] or 'N/A'}")
        lines.append("Columns:")
        for _, _, col, dtype, cdesc, _, _ in cols:
            lines.append(f"  - {col} ({dtype}) | {cdesc or 'N/A'}")
        lines.append("")

    return "\n".join(lines)


def generate_sql(client: Client, model: str, question: str, schema_text: str) -> str:
    prompt = f"""
You are a PostgreSQL expert.
Generate exactly one safe, read-only SQL query for the question.
Rules:
- Return ONLY SQL, no markdown.
- Allowed statements: SELECT or WITH ... SELECT.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, REVOKE, COPY.
- Use LIMIT when user asks for top/listing where appropriate.

Schema:
{schema_text}

Question:
{question}
""".strip()

    response = client.generate(model=model, prompt=prompt, options={"temperature": 0})
    sql = response["response"].strip()
    return sql


def validate_read_only_sql(sql: str) -> tuple[bool, str]:
    parsed = sqlparse.parse(sql)
    if not parsed:
        return False, "No SQL generated."
    if len(parsed) > 1:
        return False, "Multiple SQL statements are not allowed."

    normalized = sqlparse.format(sql, keyword_case="upper", strip_comments=True).strip()
    banned = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "GRANT", "REVOKE", "COPY"]
    for token in banned:
        if f" {token} " in f" {normalized} ":
            return False, f"Blocked potentially destructive statement: {token}"

    first = parsed[0].token_first(skip_cm=True)
    if not first:
        return False, "Unable to parse SQL statement."

    first_value = first.value.upper()
    if first_value not in {"SELECT", "WITH"}:
        return False, f"Only SELECT/WITH queries are allowed, got: {first_value}"

    return True, "OK"


def run_query(conn: psycopg.Connection, sql: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql)
        if cur.description is None:
            return pd.DataFrame()
        cols = [desc.name for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def explain_results(client: Client, model: str, question: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows returned for this question."

    sample_csv = df.head(20).to_csv(index=False)
    prompt = f"""
You are a data analyst assistant.
Given the user's question and SQL result sample, explain the result in plain English for non-technical users.
Keep it concise (3-5 sentences).

Question: {question}
Result sample:
{sample_csv}
""".strip()

    response = client.generate(model=model, prompt=prompt, options={"temperature": 0.2})
    return response["response"].strip()


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def main() -> None:
    st.set_page_config(page_title="SQLCoder Reports", page_icon="🧠", layout="wide")
    st.title("🧠 Natural Language to PostgreSQL (SQLCoder + Ollama)")
    st.caption("Ask questions in plain English. The app generates SQL, executes it safely (read-only), and explains the results.")

    init_state()

    with st.sidebar:
        st.header("Configuration")
        db_cfg = get_db_config()
        model_cfg = get_model_config()

        db_schema = st.text_input("DB schema", value=os.getenv("PGSCHEMA", "public"))
        show_sql = st.toggle("Show generated SQL", value=True)

        st.markdown("**PostgreSQL**")
        st.code(f"{db_cfg.user}@{db_cfg.host}:{db_cfg.port}/{db_cfg.dbname}")
        st.markdown("**Ollama**")
        st.code(f"{model_cfg.host} | model: {model_cfg.model}")

    question = st.text_input("Enter your question", placeholder="e.g., List top 5 employees with highest salary")

    if st.button("Submit", type="primary") and question.strip():
        try:
            ollama_client = Client(host=model_cfg.host)
            with get_connection(db_cfg) as conn:
                with st.spinner("Reading schema metadata..."):
                    schema_text = schema_context(conn, schema=db_schema)

                with st.spinner("Generating SQL with SQLCoder..."):
                    generated_sql = generate_sql(ollama_client, model_cfg.model, question, schema_text)

                valid, message = validate_read_only_sql(generated_sql)
                if not valid:
                    st.error(f"SQL validation failed: {message}")
                    return

                with st.spinner("Executing SQL..."):
                    df = run_query(conn, generated_sql)

                with st.spinner("Generating explanation..."):
                    explanation = explain_results(ollama_client, model_cfg.model, question, df)

            record = {
                "question": question,
                "sql": generated_sql,
                "rows": len(df),
                "explanation": explanation,
                "df": df,
            }
            st.session_state.history.insert(0, record)

            st.success("Query completed successfully.")
            if show_sql:
                st.subheader("Generated SQL")
                st.code(generated_sql, language="sql")

            st.subheader("Results")
            if df.empty:
                st.info("No rows returned.")
            else:
                st.dataframe(df, use_container_width=True)

            st.subheader("Explanation")
            st.write(explanation)

        except psycopg.Error as exc:
            st.error(f"Database error: {exc}")
        except Exception as exc:
            st.error(f"Application error: {exc}")

    if st.session_state.history:
        st.divider()
        st.subheader("Query History")
        for i, item in enumerate(st.session_state.history[:10], start=1):
            with st.expander(f"{i}. {item['question']} ({item['rows']} rows)"):
                if show_sql:
                    st.code(item["sql"], language="sql")
                st.dataframe(item["df"], use_container_width=True)
                st.write(item["explanation"])


if __name__ == "__main__":
    main()
