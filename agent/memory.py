"""
Persistent memory for the data agent (SQLite, no external services).

Design goals (see README "Memory design"):
  * Keep DISTILLED facts, never chat logs or data dumps.
  * Four kinds of memory:
        definition  - what a column/metric means      ("revenue = net of refunds")
        filter      - a reusable named pandas filter   ("active_customers" -> status != 'cancelled')
        preference  - how the user wants answers       ("time_granularity" -> W)
        correction  - a fact the user corrected        ("the 'amt' column is in cents")
  * Scoped: preferences are GLOBAL; everything else is scoped to a dataset
    "fingerprint" (hash of the column names) so a definition about one dataset
    is never applied to a different one.
  * Conflicts: saving a fact whose key already exists SUPERSEDES the old one
    (old row is kept as history, status='superseded') instead of piling up
    contradictions.
  * Retrieval: small BM25-style relevance score over key+content, with
    preferences always injected and filters/definitions injected when relevant.
  * Control: list / forget / clear are exposed to the user (CLI + UI).
"""
from __future__ import annotations

import difflib
import hashlib
import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

KINDS = ("definition", "filter", "preference", "correction")
GLOBAL_SCOPE = "global"
MAX_CONTENT_CHARS = 300  # a memory is a fact, not a document

_STOP = {
    "the", "a", "an", "of", "is", "are", "to", "in", "on", "for", "and", "or", "by",
    "what", "how", "me", "my", "show", "give", "with", "all", "does", "do", "be",
    "that", "this", "it", "as", "at", "from", "can", "you", "please",
}


def dataset_fingerprint(columns) -> str:
    """Datasets with the same column set share memory scope."""
    joined = "|".join(sorted(str(c).strip().lower() for c in columns))
    return "ds_" + hashlib.sha1(joined.encode()).hexdigest()[:10]


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _stem(tok: str) -> str:
    for suf in ("ing", "ers", "er", "ies", "es", "s", "ed", "ly"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(t) for t in toks if t not in _STOP and len(t) > 1]


class MemoryStore:
    def __init__(self, path: str | Path = "memory.db"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                kind TEXT NOT NULL,
                key TEXT NOT NULL,
                content TEXT NOT NULL,
                expression TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                supersedes INTEGER,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                use_count INTEGER NOT NULL DEFAULT 0
            )"""
        )
        self.conn.commit()

    # ------------------------------------------------------------------ write
    def validate(self, kind: str, key: str, content: str) -> str | None:
        """Gatekeeper: return an error string if this is not worth remembering."""
        if kind not in KINDS:
            return f"kind must be one of {KINDS}"
        if not normalize_key(key):
            return "key is empty"
        if not content.strip():
            return "content is empty"
        if len(content) > MAX_CONTENT_CHARS:
            return f"content too long ({len(content)} chars > {MAX_CONTENT_CHARS}); store a short distilled fact, not data"
        digits = sum(ch.isdigit() for ch in content)
        if digits > 0.5 * len(content) and len(content) > 40:
            return "looks like raw numbers/data, not a reusable fact"
        return None

    def _find_conflict(self, scope: str, kind: str, key: str):
        nkey = normalize_key(key)
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE scope=? AND kind=? AND status='active'",
            (scope, kind),
        ).fetchall()
        # exact match first, then fuzzy match on the key ("active_customer" ~ "active_customers")
        for r in rows:
            if r["key"] == nkey:
                return r
        best, best_ratio = None, 0.0
        for r in rows:
            ratio = difflib.SequenceMatcher(None, r["key"], nkey).ratio()
            if ratio > best_ratio:
                best, best_ratio = r, ratio
        return best if best_ratio >= 0.88 else None

    def save(self, kind: str, key: str, content: str, expression: str | None = None,
             scope: str = GLOBAL_SCOPE) -> dict:
        err = self.validate(kind, key, content)
        if err:
            return {"saved": False, "reason": err}
        if kind == "preference":
            scope = GLOBAL_SCOPE  # preferences follow the user across datasets
        nkey = normalize_key(key)
        now = time.time()
        old = self._find_conflict(scope, kind, nkey)
        if old is not None:
            same = old["content"].strip() == content.strip() and (old["expression"] or "") == (expression or "")
            if same:
                return {"saved": False, "reason": "already remembered", "id": old["id"]}
            self.conn.execute("UPDATE memories SET status='superseded', updated_at=? WHERE id=?", (now, old["id"]))
        cur = self.conn.execute(
            "INSERT INTO memories(scope,kind,key,content,expression,supersedes,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (scope, kind, nkey, content.strip(), expression, old["id"] if old else None, now, now),
        )
        self.conn.commit()
        out = {"saved": True, "id": cur.lastrowid, "key": nkey, "kind": kind}
        if old is not None:
            out["replaced"] = {"id": old["id"], "old_content": old["content"], "old_expression": old["expression"]}
        return out

    def forget(self, memory_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def clear(self, scope: str | None = None) -> int:
        if scope:
            cur = self.conn.execute("DELETE FROM memories WHERE scope=?", (scope,))
        else:
            cur = self.conn.execute("DELETE FROM memories")
        self.conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------- read
    def list(self, scope: str | None = None, include_history: bool = False) -> list[dict]:
        q = "SELECT * FROM memories WHERE 1=1"
        args: list = []
        if not include_history:
            q += " AND status='active'"
        if scope:
            q += " AND (scope=? OR scope=?)"
            args += [scope, GLOBAL_SCOPE]
        q += " ORDER BY updated_at DESC"
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def get_preference(self, key: str, default: str | None = None) -> str | None:
        r = self.conn.execute(
            "SELECT content FROM memories WHERE scope=? AND kind='preference' AND key=? AND status='active'",
            (GLOBAL_SCOPE, normalize_key(key)),
        ).fetchone()
        return r["content"] if r else default

    def get_filter(self, name: str, scope: str) -> dict | None:
        nkey = normalize_key(name)
        for s in (scope, GLOBAL_SCOPE):
            r = self.conn.execute(
                "SELECT * FROM memories WHERE scope=? AND kind='filter' AND key=? AND status='active'",
                (s, nkey),
            ).fetchone()
            if r:
                return dict(r)
        return None

    def retrieve(self, query: str, scope: str, k: int = 6) -> list[dict]:
        """Preferences are always returned; other memories only if relevant to the query."""
        rows = self.list(scope=scope)
        if not rows:
            return []
        docs = [tokenize(f"{r['key'].replace('_', ' ')} {r['content']} {r['expression'] or ''}") for r in rows]
        qtoks = tokenize(query)
        n = len(docs)
        avgdl = sum(len(d) for d in docs) / max(n, 1) or 1.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        chosen: list[tuple[float, dict]] = []
        for r, d in zip(rows, docs):
            if r["kind"] == "preference":
                chosen.append((1e6, r))
                continue
            tf = Counter(d)
            score = 0.0
            for t in set(qtoks):
                if t not in tf:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * (tf[t] * 2.2) / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avgdl))
            # key matches are strong evidence the user is naming this concept
            key_toks = set(tokenize(r["key"].replace("_", " ")))
            if key_toks and key_toks <= set(qtoks):
                score += 2.0
            if score > 0:
                chosen.append((score, r))
        chosen.sort(key=lambda x: -x[0])
        out = [r for _, r in chosen[:k]]
        ids = [r["id"] for r in out]
        if ids:
            self.conn.execute(
                f"UPDATE memories SET use_count=use_count+1 WHERE id IN ({','.join('?'*len(ids))})", ids
            )
            self.conn.commit()
        return out

    @staticmethod
    def format_for_prompt(memories: list[dict]) -> str:
        if not memories:
            return "(no stored memories are relevant to this question)"
        lines = []
        for m in memories:
            if m["kind"] == "filter":
                lines.append(f"- [filter] '{m['key']}': {m['content']}  => pandas expression: {m['expression']}")
            elif m["kind"] == "preference":
                lines.append(f"- [preference] {m['key']} = {m['content']}")
            else:
                lines.append(f"- [{m['kind']}] {m['key']}: {m['content']}")
        return "\n".join(lines)
