"""PostgreSQL connection management."""

import psycopg2
from psycopg2.extensions import connection as Connection
from typing import Optional

from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("database")


def get_db_connection() -> Connection:
    """
    Create a PostgreSQL connection.

    Returns:
        psycopg2 connection object
    """
    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise


def init_database():
    """
    Initialize database with pgvector extension and schema.

    This should be run once after database creation.
    """
    logger.info("Initializing database...")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Read and execute schema
        from pathlib import Path
        schema_path = Path(__file__).parent / "init_db.sql"

        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        cursor.execute(schema_sql)
        conn.commit()

        cursor.close()
        conn.close()

        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise


def test_connection() -> bool:
    """
    Test database connection.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False
