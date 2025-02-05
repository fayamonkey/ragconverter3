from typing import Dict, Any, List, Optional
import logging
from sentence_transformers import SentenceTransformer
import pinecone
import weaviate
import chromadb
import os
import json

class VectorExporter:
    """Handles export to various vector databases."""
    
    def __init__(self):
        self.model = None
        self.supported_dbs = ['pinecone', 'weaviate', 'chroma']
        
    def _ensure_model_loaded(self):
        """Load the embedding model if not already loaded."""
        if self.model is None:
            # Using a multilingual model to support multiple languages
            self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        self._ensure_model_loaded()
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def export_to_pinecone(self, chunks: List[Dict[str, Any]], api_key: str, 
                          environment: str, index_name: str) -> Dict[str, Any]:
        """Export chunks to Pinecone."""
        try:
            pinecone.init(api_key=api_key, environment=environment)
            
            # Create index if it doesn't exist
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(index_name, dimension=768)  # mpnet dimension
            
            index = pinecone.Index(index_name)
            
            # Prepare vectors
            texts = [chunk['text'] for chunk in chunks]
            vectors = self._get_embeddings(texts)
            
            # Prepare metadata
            metadata = []
            for chunk in chunks:
                meta = {
                    'filename': chunk['metadata']['filename'],
                    'type': chunk['metadata']['type'],
                    'start': chunk['start'],
                    'end': chunk['end']
                }
                if 'language' in chunk['metadata']:
                    meta['language'] = chunk['metadata']['language']
                metadata.append(meta)
            
            # Upload in batches
            batch_size = 100
            total_uploaded = 0
            
            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size]
                
                vectors_with_ids = list(zip(
                    [f"chunk_{j}" for j in range(i, i + len(batch_vectors))],
                    batch_vectors,
                    batch_metadata
                ))
                
                index.upsert(vectors=vectors_with_ids)
                total_uploaded += len(batch_vectors)
            
            return {
                'status': 'success',
                'database': 'pinecone',
                'chunks_uploaded': total_uploaded,
                'index': index_name
            }
            
        except Exception as e:
            logging.error(f"Pinecone export error: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def export_to_weaviate(self, chunks: List[Dict[str, Any]], url: str, 
                          api_key: Optional[str] = None) -> Dict[str, Any]:
        """Export chunks to Weaviate."""
        try:
            auth_config = weaviate.auth.AuthApiKey(api_key=api_key) if api_key else None
            
            client = weaviate.Client(
                url=url,
                auth_client_secret=auth_config,
                additional_headers={
                    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")  # Optional for hybrid search
                }
            )
            
            # Create schema if it doesn't exist
            if not client.schema.exists("RAGChunk"):
                class_obj = {
                    "class": "RAGChunk",
                    "vectorizer": "none",  # We'll provide our own vectors
                    "properties": [
                        {"name": "text", "dataType": ["text"]},
                        {"name": "filename", "dataType": ["string"]},
                        {"name": "type", "dataType": ["string"]},
                        {"name": "language", "dataType": ["string"]},
                        {"name": "start", "dataType": ["int"]},
                        {"name": "end", "dataType": ["int"]}
                    ]
                }
                client.schema.create_class(class_obj)
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            vectors = self._get_embeddings(texts)
            
            # Upload chunks
            with client.batch as batch:
                batch.batch_size = 100
                
                for chunk, vector in zip(chunks, vectors):
                    properties = {
                        "text": chunk['text'],
                        "filename": chunk['metadata']['filename'],
                        "type": chunk['metadata']['type'],
                        "start": chunk['start'],
                        "end": chunk['end']
                    }
                    if 'language' in chunk['metadata']:
                        properties["language"] = chunk['metadata']['language']
                    
                    batch.add_data_object(
                        data_object=properties,
                        class_name="RAGChunk",
                        vector=vector
                    )
            
            return {
                'status': 'success',
                'database': 'weaviate',
                'chunks_uploaded': len(chunks)
            }
            
        except Exception as e:
            logging.error(f"Weaviate export error: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def export_to_chroma(self, chunks: List[Dict[str, Any]], 
                        persist_directory: str, collection_name: str = "rag_chunks") -> Dict[str, Any]:
        """Export chunks to ChromaDB."""
        try:
            client = chromadb.PersistentClient(path=persist_directory)
            
            # Create or get collection
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Prepare data
            texts = [chunk['text'] for chunk in chunks]
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            embeddings = self._get_embeddings(texts)
            metadatas = [{
                'filename': chunk['metadata']['filename'],
                'type': chunk['metadata']['type'],
                'start': chunk['start'],
                'end': chunk['end'],
                'language': chunk['metadata'].get('language', 'unknown')
            } for chunk in chunks]
            
            # Add chunks in batches
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                collection.add(
                    ids=ids[i:i + batch_size],
                    embeddings=embeddings[i:i + batch_size],
                    documents=texts[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size]
                )
            
            return {
                'status': 'success',
                'database': 'chroma',
                'chunks_uploaded': len(chunks),
                'persist_directory': persist_directory,
                'collection_name': collection_name
            }
            
        except Exception as e:
            logging.error(f"ChromaDB export error: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_supported_databases(self) -> List[str]:
        """Get list of supported vector databases."""
        return self.supported_dbs 