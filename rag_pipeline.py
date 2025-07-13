import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import gc
import torch
import logging
from functools import lru_cache
import os
import pickle

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Disable OpenMP for FAISS to avoid threading issues
os.environ['OMP_NUM_THREADS'] = '1'

@dataclass
class RetrievedContext:
    text: str
    source: str
    similarity_score: float
    metadata: Optional[Dict[str, Any]] = None
    chunk_id: Optional[str] = None

class RAGPipeline:
    def __init__(self, data_dir: str = "data", model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Initializing RAGPipeline with data_dir={data_dir}, model={model_name}")
        self.data_dir = Path(data_dir)
        self.encoder = SentenceTransformer(model_name)
        self.encoder.to('cpu')  # Force CPU for consistent performance
        self.embedding_dim = 384
        
        # Initialize data structures
        self.chunks_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.faiss_indices: Dict[str, faiss.Index] = {}
        
        # Cache for query embeddings
        self.query_cache = {}
        self.max_cache_size = 1000
        
        # Create processed data directory if it doesn't exist
        self.processed_dir = Path("processed_data")
        self.processed_dir.mkdir(exist_ok=True)
        
        # Preload and cache everything
        self._load_data()
        
        # Warm up the encoder
        logger.info("Warming up the encoder...")
        _ = self.encoder.encode("warmup query", convert_to_tensor=True)

    def _save_faiss_index(self, index: faiss.Index, category: str):
        """Save FAISS index to disk"""
        try:
            index_path = self.processed_dir / f"{category}_faiss.index"
            logger.info(f"Saving FAISS index for {category} to {index_path}")
            faiss.write_index(index, str(index_path))
            
            # Save embeddings metadata
            metadata_path = self.processed_dir / f"{category}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'embedding_dim': self.embedding_dim,
                    'num_vectors': index.ntotal
                }, f)
            
            logger.info(f"Successfully saved FAISS index and metadata for {category}")
        except Exception as e:
            logger.error(f"Error saving FAISS index for {category}: {str(e)}", exc_info=True)
            raise

    def _load_faiss_index(self, category: str) -> Optional[faiss.Index]:
        """Load FAISS index from disk if it exists"""
        try:
            index_path = self.processed_dir / f"{category}_faiss.index"
            metadata_path = self.processed_dir / f"{category}_metadata.pkl"
            
            if not index_path.exists() or not metadata_path.exists():
                logger.info(f"No existing FAISS index found for {category}")
                return None
            
            # Load and verify metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                
            if metadata['embedding_dim'] != self.embedding_dim:
                logger.warning(f"Embedding dimension mismatch for {category}. Expected {self.embedding_dim}, got {metadata['embedding_dim']}")
                return None
            
            logger.info(f"Loading FAISS index for {category} from {index_path}")
            index = faiss.read_index(str(index_path))
            
            if index.ntotal != metadata['num_vectors']:
                logger.warning(f"Vector count mismatch for {category}. Expected {metadata['num_vectors']}, got {index.ntotal}")
                return None
                
            return index
        except Exception as e:
            logger.error(f"Error loading FAISS index for {category}: {str(e)}", exc_info=True)
            return None

    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create a FAISS index for the given embeddings."""
        try:
            logger.debug(f"Creating FAISS index for embeddings shape: {embeddings.shape}")
            
            # Convert to float32 (required by FAISS)
            embeddings = embeddings.astype('float32')
            
            # Create a simple FlatL2 index - more reliable for small datasets
            index = faiss.IndexFlatL2(self.embedding_dim)
            
            # Add vectors
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            index.add(embeddings)
            logger.debug(f"Added {len(embeddings)} vectors to index")
            
            return index
        except Exception as e:
            logger.error(f"Error in _create_faiss_index: {str(e)}", exc_info=True)
            raise

    def _save_embeddings(self, embeddings: np.ndarray, category: str):
        """Save pre-computed embeddings to disk"""
        try:
            embeddings_path = self.processed_dir / f"{category}_embeddings.npy"
            logger.info(f"Saving embeddings for {category} to {embeddings_path}")
            np.save(str(embeddings_path), embeddings)
            logger.info(f"Successfully saved embeddings for {category}")
        except Exception as e:
            logger.error(f"Error saving embeddings for {category}: {str(e)}", exc_info=True)
            raise

    def _load_embeddings(self, category: str) -> Optional[np.ndarray]:
        """Load pre-computed embeddings from disk if they exist"""
        try:
            embeddings_path = self.processed_dir / f"{category}_embeddings.npy"
            
            if not embeddings_path.exists():
                logger.info(f"No existing embeddings found for {category}")
                return None
            
            logger.info(f"Loading embeddings for {category} from {embeddings_path}")
            embeddings = np.load(str(embeddings_path))
            
            # Verify embedding dimension
            if embeddings.shape[1] != self.embedding_dim:
                logger.warning(f"Embedding dimension mismatch for {category}. Expected {self.embedding_dim}, got {embeddings.shape[1]}")
                return None
                
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings for {category}: {str(e)}", exc_info=True)
            return None

    def _compute_embeddings(self, texts: List[str], batch_size: int = 50) -> np.ndarray:
        """Compute embeddings for a list of texts in batches"""
        try:
            embeddings_list = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                with torch.no_grad():
                    batch_embeddings = self.encoder.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )
                embeddings_list.append(batch_embeddings.cpu().numpy())
                del batch_embeddings
                gc.collect()
                torch.cuda.empty_cache()
            
            # Combine all embeddings
            embeddings = np.vstack(embeddings_list)
            del embeddings_list
            gc.collect()
            
            return embeddings
        except Exception as e:
            logger.error(f"Error computing embeddings: {str(e)}", exc_info=True)
            raise

    @lru_cache(maxsize=128)
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get cached query embedding or compute new one."""
        with torch.no_grad():
            query_embedding = self.encoder.encode(query, convert_to_tensor=True)
            return query_embedding.cpu().numpy().reshape(1, -1)

    def _process_text_file(self, file_path: Path, max_chunks: int = 100) -> List[Dict[str, Any]]:  # Reduced chunks
        """Process a text file into chunks."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Split content into sections and limit size
            sections = content.split('\n\n')
            chunks = []
            
            for section in sections[:max_chunks]:
                if section.strip():
                    # Limit chunk size
                    text = section.strip()[:500]  # Limit to 500 chars
                    chunks.append({
                        'text': text,
                        'source': file_path.stem,
                        'id': f"{file_path.stem}_{len(chunks)}"
                    })
            
            logger.info(f"Processed {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            return []

    def _load_data(self):
        """Load data from text files"""
        try:
            # Process each text file
            for file_path in [self.data_dir / "common_issues.txt", self.data_dir / "linux_manual.txt"]:
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                logger.info(f"Processing file: {file_path}")
                chunks = self._process_text_file(file_path)
                if not chunks:
                    continue
                
                category = file_path.stem
                self.chunks_cache[category] = chunks
                
                # Try to load existing FAISS index and embeddings
                existing_index = self._load_faiss_index(category)
                existing_embeddings = self._load_embeddings(category)
                
                if existing_index is not None and existing_embeddings is not None:
                    self.faiss_indices[category] = existing_index
                    logger.info(f"Successfully loaded existing FAISS index and embeddings for {category}")
                    continue
                
                try:
                    # Compute embeddings
                    all_texts = [chunk['text'] for chunk in chunks]
                    embeddings_np = self._compute_embeddings(all_texts)
                    
                    # Save embeddings
                    self._save_embeddings(embeddings_np, category)
                    
                    # Create and save FAISS index
                    self.faiss_indices[category] = self._create_faiss_index(embeddings_np)
                    self._save_faiss_index(self.faiss_indices[category], category)
                    
                    del embeddings_np
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Error processing embeddings for {category}: {str(e)}", exc_info=True)
                    continue
                
        except Exception as e:
            logger.error(f"Error in _load_data: {str(e)}", exc_info=True)
            raise

    def _search_category(self, query_embedding: np.ndarray, category: str, top_k: int) -> List[RetrievedContext]:
        """Search within a specific category using FAISS."""
        try:
            # Convert query to float32 and reshape if needed
            query_embedding = query_embedding.astype('float32')
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search the FAISS index
            D, I = self.faiss_indices[category].search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx != -1:  # FAISS returns -1 for not found
                    chunk = self.chunks_cache[category][idx]
                    results.append(
                        RetrievedContext(
                            text=chunk['text'],
                            source=chunk.get('source', category),
                            similarity_score=float(score),
                            metadata=chunk.get('metadata', {}),
                            chunk_id=chunk.get('id', f"{category}_{idx}")
                        )
                    )
            return results
            
        except Exception as e:
            logger.error(f"Error searching category {category}: {str(e)}", exc_info=True)
            return []

    def retrieve(self, query: str, top_k: int = 3, quality_threshold: float = 1.5) -> List[RetrievedContext]:
        """
        Retrieve relevant context for the given query, prioritizing linux_manual over common_issues.
        
        Args:
            query: User query string
            top_k: Number of most relevant chunks to return per category
            quality_threshold: Similarity score threshold below which to check common_issues
            
        Returns:
            List of RetrievedContext objects containing relevant information
        """
        try:
            # Get cached or compute query embedding
            query_embedding_np = self._get_query_embedding(query)
            
            # First search linux_manual
            manual_results = []
            if 'linux_manual' in self.chunks_cache:
                manual_results = self._search_category(query_embedding_np, 'linux_manual', top_k)
            
            # Check if manual results are good enough
            if manual_results and manual_results[0].similarity_score <= quality_threshold:
                # If manual results aren't good enough, search common_issues
                if 'common_issues' in self.chunks_cache:
                    issue_results = self._search_category(query_embedding_np, 'common_issues', top_k)
                    # Combine and sort all results
                    all_results = manual_results + issue_results
                    all_results.sort(key=lambda x: x.similarity_score, reverse=True)
                    return all_results[:top_k]
            
            return manual_results[:top_k] if manual_results else []
            
        except Exception as e:
            logger.error(f"Error in retrieve: {str(e)}", exc_info=True)
            return []

def main():
    # Example usage
    pipeline = RAGPipeline()
    
    test_queries = [
        "How do I check disk space usage?",
        "What is the command to create a new directory?",
        "How to fix permission denied errors?",
        "Show me how to use the ls command"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = pipeline.retrieve(query)
        print(f"Top {len(results)} relevant chunks:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.similarity_score:.3f}")
            print(f"Source: {result.source}")
            print(f"Chunk ID: {result.chunk_id}")
            print(f"Text: {result.text[:200]}...")
            print(f"Using fallback: {result.source == 'common_issues'}")

if __name__ == "__main__":
    main() 