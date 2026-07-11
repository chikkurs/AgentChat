from typing import List, Tuple

from db import get_db_connection


def save_message(
    user_id: str,
    role: str,
    message: str,
) -> None:
    """
    Save one user or assistant message in PostgreSQL.
    """

    if role not in {"user", "assistant"}:
        raise ValueError("Role must be 'user' or 'assistant'")

    if not message or not message.strip():
        return

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO chat_memory (
                user_id,
                role,
                message
            )
            VALUES (%s, %s, %s)
            """,
            (
                str(user_id),
                role,
                message.strip(),
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


def get_chat_history(
    user_id: str,
    limit: int = 10,
) -> List[Tuple[str, str]]:
    """
    Return the latest messages in chronological order.
    """

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT role, message
            FROM (
                SELECT
                    id,
                    role,
                    message,
                    created_at
                FROM chat_memory
                WHERE user_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
            ) AS recent_messages
            ORDER BY created_at ASC, id ASC
            """,
            (
                str(user_id),
                limit,
            ),
        )

        rows = cur.fetchall()

        return rows

    finally:
        if cur:
            cur.close()

        if conn:
            conn.close()


def format_chat_history(
    history: List[Tuple[str, str]],
) -> str:
    """
    Convert database messages into prompt text.
    """

    if not history:
        return "No previous conversation."

    formatted_messages = []

    for role, message in history:
        display_role = (
            "User"
            if role == "user"
            else "Assistant"
        )

        formatted_messages.append(
            f"{display_role}: {message}"
        )

    return "\n".join(formatted_messages)


def clear_chat_history(user_id: str) -> int:
    """
    Delete memory belonging to one user.
    Returns the number of deleted rows.
    """

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            DELETE FROM chat_memory
            WHERE user_id = %s
            """,
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