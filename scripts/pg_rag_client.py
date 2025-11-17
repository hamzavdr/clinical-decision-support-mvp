import os
from typing import Any, Dict, Iterable, List, Optional

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer


def _vector_literal(vec: Iterable[float]) -> str:
    parts = []
    for v in vec:
        s = f"{v:.8f}".rstrip("0").rstrip(".")
        parts.append(s or "0")
    return "[" + ", ".join(parts) + "]"


def _build_conninfo(cfg: Optional[Dict[str, Any]] = None) -> str:
    cfg = cfg or {}
    env_conn = os.getenv("PG_CONNINFO") or cfg.get("conninfo")
    if env_conn:
        return env_conn

    host = os.getenv("PGHOST") or cfg.get("host")
    dbname = os.getenv("PGDATABASE") or cfg.get("db")
    user = os.getenv("PGUSER") or cfg.get("user")
    password = os.getenv("PGPASSWORD") or cfg.get("password")
    port = os.getenv("PGPORT") or cfg.get("port", 5432)
    if not all([host, dbname, user, password]):
        raise RuntimeError("Missing Postgres connection info. Set PG* env vars or pg.conninfo in config.")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


class PgVectorRAGClient:
    def __init__(self, pg_cfg: Optional[Dict[str, Any]], embed_model: str, table: str):
        conninfo = _build_conninfo(pg_cfg)
        self.table = table
        self.conn = psycopg.connect(conninfo, row_factory=dict_row)
        self.conn.autocommit = True
        self.embedder = SentenceTransformer(embed_model, device="cpu")
        print(f"[PgVectorRAGClient] Connected to Postgres table '{table}'")

    def _run_query(self, query: sql.SQL, params: tuple):
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def query(self, q: str, doc_id: str, k: int = 4):
        print(f"[PgVectorRAGClient.query] Query: '{q[:50]}...', doc_id: '{doc_id}', k: {k}")
        vec = _vector_literal(self.embedder.encode([q])[0])
        query_sql = sql.SQL(
            """
            SELECT id, doc_id, window_id, page_start, page_end, text_chunk,
                   embedding <=> %s::vector AS distance
            FROM {table}
            WHERE doc_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
        ).format(table=sql.Identifier(self.table))
        rows = self._run_query(query_sql, (vec, doc_id, vec, k))
        hits = []
        for row in rows:
            meta = {
                "doc_id": row["doc_id"],
                "window_id": row["window_id"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
            }
            hits.append(
                {
                    "id": row["id"],
                    "document": row["text_chunk"],
                    "metadata": meta,
                    "distance": float(row["distance"]),
                }
            )
        return hits

    def fetch(self, doc_id: str, window_id: int):
        query_sql = sql.SQL(
            """
            SELECT text_chunk, doc_id, window_id, page_start, page_end
            FROM {table}
            WHERE doc_id = %s AND window_id = %s
            LIMIT 1
            """
        ).format(table=sql.Identifier(self.table))
        rows = self._run_query(query_sql, (doc_id, window_id))
        if not rows:
            return None, None
        row = rows[0]
        meta = {
            "doc_id": row["doc_id"],
            "window_id": row["window_id"],
            "page_start": row["page_start"],
            "page_end": row["page_end"],
        }
        return row["text_chunk"], meta

    def stitch_neighbors(self, hit: Dict[str, Any], above=1, below=2, cap_words=1600):
        m = hit["metadata"]
        center = int(m["window_id"])
        parts, total = [], 0
        for w in range(center - above, center + below + 1):
            doc, meta = self.fetch(m["doc_id"], w)
            if doc is None:
                continue
            parts.append((doc, meta))
            total += len(doc.split())
            if total >= cap_words:
                break
        if not parts:
            return None
        text = "\n".join(p[0] for p in parts)
        page_start = parts[0][1].get("page_start", m.get("page_start", 0))
        page_end = parts[-1][1].get("page_end", m.get("page_end", 0))
        return {"text": text, "pages": (int(page_start), int(page_end)), "windows": [p[1]["window_id"] for p in parts]}
