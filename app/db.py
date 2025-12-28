import chromadb


class ChromaStore:
    def __init__(self, path: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, documents, embeddings, metadatas, ids):
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query_embedding, top_k: int):
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
        )
        
        return (
            results["documents"][0],
            results["metadatas"][0],
        )

