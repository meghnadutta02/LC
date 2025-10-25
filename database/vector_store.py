"""
Vector store operations using pgvector.

Handles document storage with OpenAI embeddings and semantic search.
"""

import uuid
from typing import List, Dict, Any, Optional
from openai import OpenAI
import numpy as np

from database.connection import get_db_connection
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("vector_store")


class VectorStore:
    """
    Vector store for document embeddings using pgvector.

    Uses OpenAI text-embedding-3-small (1536 dimensions) for embeddings.
    """

    def __init__(self):
        """Initialize vector store with OpenAI client."""
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.embedding_model = settings.openai_embedding_model
        logger.info(
            f"VectorStore initialized with model: {self.embedding_model}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            List of 1536 floats (embedding vector)
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text (length: {len(text)})")
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def store_document(
        self,
        workflow_id: str,
        content: str,
        title: Optional[str] = None,
        source_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store a document with its embedding in pgvector.

        Args:
            workflow_id: Workflow UUID
            content: Document content
            title: Document title
            source_url: Source URL
            metadata: Additional metadata

        Returns:
            Document ID
        """
        logger.info(f"Storing document for workflow {workflow_id}")

        try:
            # Generate embedding
            embedding = self.get_embedding(content)

            # Convert embedding to PostgreSQL array format
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            # Store in database
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO documents (workflow_id, content, title, source_url, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                RETURNING id
            """, (
                workflow_id,
                content,
                title,
                source_url,
                metadata or {},
                embedding_str
            ))

            doc_id = cursor.fetchone()[0]
            conn.commit()

            cursor.close()
            conn.close()

            logger.info(f"Document stored with ID: {doc_id}")
            return doc_id

        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            raise

    async def search(
        self,
        query: str,
        max_results: int = 10,
        similarity_threshold: float = 0.7,
        workflow_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity.

        Args:
            query: Search query
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            workflow_id: Optional workflow ID to filter results

        Returns:
            List of matching documents with similarity scores
        """
        logger.info(f"Searching for: {query}")

        try:
            # Generate query embedding
            query_embedding = self.get_embedding(query)
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # Build query
            conn = get_db_connection()
            cursor = conn.cursor()

            if workflow_id:
                sql = """
                    SELECT 
                        id,
                        content,
                        title,
                        source_url,
                        metadata,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM documents
                    WHERE workflow_id = %s
                        AND 1 - (embedding <=> %s::vector) > %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                cursor.execute(sql, (
                    embedding_str,
                    workflow_id,
                    embedding_str,
                    similarity_threshold,
                    embedding_str,
                    max_results
                ))
            else:
                sql = """
                    SELECT 
                        id,
                        content,
                        title,
                        source_url,
                        metadata,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM documents
                    WHERE 1 - (embedding <=> %s::vector) > %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                cursor.execute(sql, (
                    embedding_str,
                    embedding_str,
                    similarity_threshold,
                    embedding_str,
                    max_results
                ))

            # Fetch results
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "title": row[2],
                    "source_url": row[3],
                    "metadata": row[4],
                    "similarity": float(row[5])
                })

            cursor.close()
            conn.close()

            logger.info(f"Found {len(results)} matching documents")
            return results

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise

    async def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document data or None if not found
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, workflow_id, content, title, source_url, metadata, created_at
                FROM documents
                WHERE id = %s
            """, (doc_id,))

            row = cursor.fetchone()

            cursor.close()
            conn.close()

            if row:
                return {
                    "id": row[0],
                    "workflow_id": row[1],
                    "content": row[2],
                    "title": row[3],
                    "source_url": row[4],
                    "metadata": row[5],
                    "created_at": row[6]
                }

            return None

        except Exception as e:
            logger.error(f"Error retrieving document: {str(e)}")
            raise

    async def delete_documents(self, workflow_id: str) -> int:
        """
        Delete all documents for a workflow.

        Args:
            workflow_id: Workflow UUID

        Returns:
            Number of documents deleted
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM documents
                WHERE workflow_id = %s
            """, (workflow_id,))

            deleted_count = cursor.rowcount
            conn.commit()

            cursor.close()
            conn.close()

            logger.info(
                f"Deleted {deleted_count} documents for workflow {workflow_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise

    async def get_similar_documents(
        self,
        doc_id: int,
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document.

        Args:
            doc_id: Reference document ID
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity score

        Returns:
            List of similar documents
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Get the reference document's embedding
            cursor.execute("""
                SELECT embedding FROM documents WHERE id = %s
            """, (doc_id,))

            result = cursor.fetchone()
            if not result:
                return []

            ref_embedding = result[0]

            # Find similar documents
            cursor.execute("""
                SELECT 
                    id,
                    content,
                    title,
                    source_url,
                    metadata,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                WHERE id != %s
                    AND 1 - (embedding <=> %s::vector) > %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (
                ref_embedding,
                doc_id,
                ref_embedding,
                similarity_threshold,
                ref_embedding,
                max_results
            ))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "title": row[2],
                    "source_url": row[3],
                    "metadata": row[4],
                    "similarity": float(row[5])
                })

            cursor.close()
            conn.close()

            return results

        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            raise
