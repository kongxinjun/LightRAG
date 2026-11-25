"""Backend combinations the harness can drive.

Each profile is (storage class names, environment). Docker profiles point at
the services in ``docker-compose.hunt.yml``.
"""

from __future__ import annotations

DOCKER_ENV = {
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "15432",
    "POSTGRES_USER": "hunt",
    "POSTGRES_PASSWORD": "hunt",
    "POSTGRES_DATABASE": "rag",
    "POSTGRES_VECTOR_INDEX_TYPE": "HNSW",
    "NEO4J_URI": "neo4j://localhost:17687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "huntpassword",
    "NEO4J_DATABASE": "neo4j",
    "MEMGRAPH_URI": "bolt://localhost:17688",
    "REDIS_URI": "redis://localhost:16379",
    "MONGO_URI": "mongodb://localhost:27018/?directConnection=true",
    "MONGO_DATABASE": "LightRAGHunt",
    "QDRANT_URL": "http://localhost:16333",
    "MILVUS_URI": "http://localhost:19531",
    "MILVUS_DB_NAME": "default",
}

PROFILES: dict[str, dict[str, str]] = {
    "file": dict(
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
    ),
    "file-faiss": dict(
        kv_storage="JsonKVStorage",
        vector_storage="FaissVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
    ),
    "pg": dict(
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="PGGraphStorage",
        doc_status_storage="PGDocStatusStorage",
    ),
    "pg-table": dict(
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="PGTableGraphStorage",
        doc_status_storage="PGDocStatusStorage",
    ),
    "neo4j-redis-qdrant": dict(
        kv_storage="RedisKVStorage",
        vector_storage="QdrantVectorDBStorage",
        graph_storage="Neo4JStorage",
        doc_status_storage="RedisDocStatusStorage",
    ),
    "mongo": dict(
        kv_storage="MongoKVStorage",
        vector_storage="MongoVectorDBStorage",
        graph_storage="MongoGraphStorage",
        doc_status_storage="MongoDocStatusStorage",
    ),
    "milvus": dict(
        kv_storage="JsonKVStorage",
        vector_storage="MilvusVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
    ),
    "memgraph": dict(
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="MemgraphStorage",
        doc_status_storage="JsonDocStatusStorage",
    ),
    "memgraph-milvus": dict(
        kv_storage="JsonKVStorage",
        vector_storage="MilvusVectorDBStorage",
        graph_storage="MemgraphStorage",
        doc_status_storage="JsonDocStatusStorage",
    ),
}
