import json

from typing import Dict, List, Tuple

from db import get_db_connection


# =====================================================
# SCHEMA (run once)
# =====================================================
#
# CREATE TABLE user_memory (
#     id SERIAL PRIMARY KEY,
#     user_id TEXT NOT NULL,
#     category TEXT NOT NULL,
#     entry TEXT NOT NULL,
#     updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
#     UNIQUE (user_id, category)
# );
#
# One row per (user_id, category) — e.g. a user has at most one "name"
# entry, one "communication_style" entry, etc. New extractions
# overwrite the old value for that category rather than appending, so
# the table stays small and current instead of growing forever like a
# transcript would.


def get_user_memory(user_id: str) -> List[Tuple[str, str]]:
    """
    Return all stored (category, entry) pairs for a user.
    """

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT category, entry
            FROM user_memory
            WHERE user_id = %s
            ORDER BY category ASC
            """,
            (str(user_id),),
        )

        return cur.fetchall()

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def format_memory_for_prompt(memory_rows: List[Tuple[str, str]]) -> str:
    """
    Render stored memory as a short bullet list for the system prompt.
    """

    if not memory_rows:
        return ""

    lines = [
        f"- {category}: {entry}"
        for category, entry in memory_rows
    ]

    return "\n".join(lines)


def upsert_memory_entries(
    user_id: str,
    entries: Dict[str, str],
) -> None:
    """
    Insert or update memory entries for a user.

    entries is a dict of {category: entry_text}. Empty/blank values
    are skipped. Existing categories are overwritten with the new
    value, keeping memory current rather than accumulating duplicates.
    """

    if not entries:
        return

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        for category, entry in entries.items():
            category = str(category).strip().lower()
            entry = str(entry).strip()

            if not category or not entry:
                continue

            cur.execute(
                """
                INSERT INTO user_memory (user_id, category, entry, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (user_id, category)
                DO UPDATE SET
                    entry = EXCLUDED.entry,
                    updated_at = NOW()
                """,
                (
                    str(user_id),
                    category,
                    entry,
                ),
            )

        conn.commit()

    except Exception:
        if conn:
            conn.rollback()
        raise

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def delete_user_memory(user_id: str) -> int:
    """
    Delete all memory entries for one user. Returns rows deleted.
    """

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            "DELETE FROM user_memory WHERE user_id = %s",
            (str(user_id),),
        )

        deleted_count = cur.rowcount
        conn.commit()

        return deleted_count

    except Exception:
        if conn:
            conn.rollback()
        raise

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def delete_memory_entry(user_id: str, category: str) -> int:
    """
    Delete a single memory category for one user. Returns rows deleted.
    """

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            DELETE FROM user_memory
            WHERE user_id = %s AND category = %s
            """,
            (str(user_id), str(category).strip().lower()),
        )

        deleted_count = cur.rowcount
        conn.commit()

        return deleted_count

    except Exception:
        if conn:
            conn.rollback()
        raise

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# =====================================================
# EXTRACTION (LLM-based fact synthesis)
# =====================================================

EXTRACTION_SYSTEM_PROMPT = """You extract durable facts worth remembering \
about a user from a single message exchange, for a long-term memory system.

Rules:
- Only extract facts that are genuinely durable and useful across future \
conversations: the user's name, role/job, stated preferences (e.g. reply \
style, tone), ongoing projects, tools/technologies they use, or similar \
lasting context.
- Do NOT extract one-off questions, small talk, greetings, or anything \
that is only relevant to this single exchange.
- If nothing durable was shared, return an empty JSON object: {}
- Respond with ONLY a JSON object, no other text, no markdown fences.
- Keys are short lowercase category labels (e.g. "name", "role", \
"preference", "project", "tech_stack"). Values are short factual \
statements (a few words to one sentence).
- If a new fact updates or contradicts an existing one, prefer the new \
fact.

Existing known facts about this user (may be empty):
{existing_memory}
"""


def extract_memory_updates(
    groq_client,
    model: str,
    question: str,
    answer: str,
    existing_memory_rows: List[Tuple[str, str]],
) -> Dict[str, str]:
    """
    Ask the LLM whether this exchange contains any durable facts worth
    remembering. Returns a dict of {category: entry}, possibly empty.

    Failures here are non-fatal by design — memory extraction should
    never break the actual chat response, so callers should wrap this
    in a try/except and just skip saving on error.
    """

    existing_text = format_memory_for_prompt(existing_memory_rows) or "(none yet)"

    system_prompt = EXTRACTION_SYSTEM_PROMPT.format(
        existing_memory=existing_text
    )

    completion = groq_client.chat.completions.create(
        model=model,
        max_tokens=300,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"User message: {question}\n\n"
                    f"Assistant reply: {answer}"
                ),
            },
        ],
    )

    raw_output = completion.choices[0].message.content or "{}"
    raw_output = raw_output.strip()

    # Strip accidental markdown fences if the model adds them anyway.
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`")
        if raw_output.lower().startswith("json"):
            raw_output = raw_output[4:].strip()

    try:
        parsed = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return {}

    if not isinstance(parsed, dict):
        return {}

    return {
        str(key): str(value)
        for key, value in parsed.items()
        if value
    }
