from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from collections import defaultdict
import logging

class DocumentStructurer:
    """Intelligent document organization and structuring."""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.topic_model = LatentDirichletAllocation(
            n_components=5,  # Default number of topics
            random_state=42
        )
    
    def _ensure_model_loaded(self):
        """Load the embedding model if not already loaded."""
        if self.model is None:
            self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    def _get_document_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Generate embeddings for documents."""
        self._ensure_model_loaded()
        texts = [doc['content'] for doc in documents]
        return self.model.encode(texts)
    
    def _extract_topics(self, documents: List[Dict[str, Any]]) -> List[List[str]]:
        """Extract main topics from documents using LDA."""
        texts = [doc['content'] for doc in documents]
        
        # Create vectorizer with parameters based on document count
        doc_count = len(texts)
        self.vectorizer = CountVectorizer(
            max_df=1.0,  # Don't ignore any words based on document frequency
            min_df=1,    # Include words that appear in at least 1 document
            stop_words='english'
        )
        
        # Create document-term matrix
        try:
            dtm = self.vectorizer.fit_transform(texts)
            
            # Adjust number of topics based on document count
            n_topics = min(self.topic_model.n_components, max(2, doc_count - 1))
            self.topic_model.n_components = n_topics
            
            # Fit LDA model
            self.topic_model.fit(dtm)
            
            # Get feature names (words)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Extract top words for each topic
            topics = []
            for topic_idx, topic in enumerate(self.topic_model.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
                topics.append(top_words)
            
            return topics
        except Exception as e:
            logging.warning(f"Topic extraction warning: {str(e)}")
            # Return basic topics based on document titles if topic modeling fails
            return [[doc['metadata']['filename']] for doc in documents]
    
    def _create_hierarchy(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> Dict[str, Any]:
        """Create hierarchical structure of documents using clustering."""
        # Handle single document case
        if len(documents) == 1:
            return {"cluster_0": [documents[0]]}
            
        # Perform hierarchical clustering for multiple documents
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,  # Adjust based on desired granularity
            metric='cosine',
            linkage='average'
        )
        clusters = clustering.fit_predict(embeddings)
        
        # Create hierarchy
        hierarchy = defaultdict(list)
        for doc_idx, cluster_id in enumerate(clusters):
            hierarchy[f"cluster_{cluster_id}"].append(documents[doc_idx])
        
        return dict(hierarchy)
    
    def _create_document_graph(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> nx.Graph:
        """Create a graph of document relationships."""
        graph = nx.Graph()
        
        # Add nodes
        for i, doc in enumerate(documents):
            graph.add_node(i, **doc['metadata'])
        
        # Add edges based on similarity (skip for single document)
        if len(documents) > 1:
            for i in range(len(documents)):
                for j in range(i + 1, len(documents)):
                    similarity = np.dot(embeddings[i], embeddings[j])
                    if similarity > 0.5:  # Adjust threshold as needed
                        graph.add_edge(i, j, weight=similarity)
        
        return graph
    
    def structure_documents(self, documents: List[Dict[str, Any]], 
                          min_similarity: float = 0.5,
                          max_topics: int = 5) -> Dict[str, Any]:
        """
        Structure and organize documents based on content similarity and topics.
        
        Args:
            documents: List of processed documents with content and metadata
            min_similarity: Minimum similarity threshold for grouping
            max_topics: Maximum number of topics to extract
            
        Returns:
            Dictionary containing structured document information
        """
        try:
            # Update topic model parameters
            self.topic_model.n_components = max_topics
            
            # Generate document embeddings
            embeddings = self._get_document_embeddings(documents)
            
            # Extract topics
            topics = self._extract_topics(documents)
            
            # Create hierarchy
            hierarchy = self._create_hierarchy(documents, embeddings)
            
            # Create document graph
            graph = self._create_document_graph(documents, embeddings)
            
            # Find central documents (highest degree centrality)
            centrality = nx.degree_centrality(graph)
            central_docs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            
            # Assign topics to clusters
            cluster_topics = {}
            for cluster_id, cluster_docs in hierarchy.items():
                cluster_texts = [doc['content'] for doc in cluster_docs]
                dtm = self.vectorizer.transform(cluster_texts)
                topic_dist = self.topic_model.transform(dtm).mean(axis=0)
                main_topic_idx = topic_dist.argmax()
                cluster_topics[cluster_id] = topics[main_topic_idx]
            
            # Create final structure
            structure = {
                'topics': topics,
                'clusters': {
                    cluster_id: {
                        'documents': cluster_docs,
                        'main_topic': cluster_topics[cluster_id],
                        'size': len(cluster_docs)
                    }
                    for cluster_id, cluster_docs in hierarchy.items()
                },
                'central_documents': [
                    documents[doc_id] for doc_id, _ in central_docs[:3]  # Top 3 central documents
                ],
                'relationships': [
                    {
                        'source': documents[source]['metadata']['filename'],
                        'target': documents[target]['metadata']['filename'],
                        'similarity': data['weight']
                    }
                    for source, target, data in graph.edges(data=True)
                ]
            }
            
            return {
                'status': 'success',
                'structure': structure
            }
            
        except Exception as e:
            logging.error(f"Error during document structuring: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            } 