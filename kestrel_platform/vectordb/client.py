from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class PremiumVectorDBClient:
    """
    Dedicated client wrapper to index and search high-dimensional underwriting matrices.
    """
    def __init__(self, host: str, port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "maritime_underwriting_vault"

    def recreate_collection(self, vector_size: int = 1536):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    def upsert_clause_vector(self, clause_id: str, vector: list, metadata: dict):
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=hash(clause_id),
                    vector=vector,
                    payload=metadata
                )
            ]
        )

    def query_semantic_clauses(self, vector: list, limit: int = 5) -> list:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=limit
        )
        return [{"score": r.score, "payload": r.payload} for r in results]
