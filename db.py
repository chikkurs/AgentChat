import os
import psycopg2


def get_db_connection():
    """
    Connect to the Postgres database (e.g. AWS RDS).

    sslmode="require" is set explicitly because RDS Postgres enforces
    SSL by default — without this, some network paths reject the
    connection with a "no pg_hba.conf entry ... no encryption" error
    even when the password itself is correct.
    """

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode="require",
        connect_timeout=10,
    )


def check_db_connection() -> None:
    """
    Test the database connection once at startup and print a clear
    success/failure message to the logs. This surfaces connection
    problems immediately in your AWS logs (CloudWatch, journalctl,
    docker logs, etc.) instead of only showing up later as silent
    per-request failures inside try/except blocks.
    """

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")

    print(
        f"[DB STARTUP CHECK] Attempting connection to "
        f"host={host} port={port} database={database} user={user} ..."
    )

    conn = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        print("[DB STARTUP CHECK] SUCCESS — database connection is working.")

    except Exception as exc:
        print(
            "[DB STARTUP CHECK] FAILED — the app cannot reach the "
            f"database. Error: {exc}"
        )
        print(
            "[DB STARTUP CHECK] Check DB_HOST, DB_PORT, DB_NAME, "
            "DB_USER, DB_PASSWORD env vars, and the RDS security "
            "group's inbound rules."
        )

    finally:
        if conn:
            conn.close()
