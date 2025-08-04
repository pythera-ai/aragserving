from trism import TritonModel
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from fastapi import HTTPException

from src.utils.utils import prepare_text_input, extract_text_from_file, generate_md5_hash

# logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.models = {}
        self._initialize_models()
        self.context_embedding_length = model_config.get("retrieve_context", {}).get("max_length", 512) # model_max_length of context embedding model

    def _initialize_models(self):
        """Initialize all Triton models"""
        try:
            for model_key, config in self.model_config.items():
                logger.info(f"Initializing {model_key} model...")
                self.models[model_key] = TritonModel(
                    model=config["name"],
                    version=config["version"],
                    url=config["url"],
                    grpc=config["grpc"]
                )
                logger.info(f"Successfully initialized {model_key} model")
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")
    
    def segment_text(self, text: str) -> List[str]:
        """Use SAT model to segment text into chunks"""
        try:
            input_text = [text.encode("utf-8")]
            input_array = np.array(input_text, dtype=object)
            result = self.models["sat"].run(data=[input_array, np.array([], dtype=object), np.array([self.context_embedding_length], dtype=np.int64)])

            # Extract chunks from SAT model output
            return result["logits"]
           
        except Exception as e:
            logger.error(f"Text segmentation failed: {str(e)}")
            # Fallback to simple splitting
            sentences = text.split('. ')
            return [s.strip() + '.' for s in sentences if s.strip()]
    
    def get_query_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings for query text"""
        try:
            input_data = prepare_text_input(text)
            result = self.models["retrieve_query"].run(data=[input_data])
            return result["embedding"]
        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query embedding failed: {str(e)}")
    
    def get_context_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings for context text"""
        try:
            input_data = prepare_text_input(text)
            result = self.models["retrieve_context"].run(data=[input_data])
            return result["embedding"]
        except Exception as e:
            logger.error(f"Context embedding failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Context embedding failed: {str(e)}")
    
    def rerank_results(self, query_text: str, context_texts: List[str]) -> List[float]:
        """Use rerank model to score query-context pairs"""
        try:
            scores = []
            for ctx_text in context_texts:
                # Prepare input for rerank model - both query and context as text
                query_input = [query_text.encode("utf-8")]
                query_input = np.array(query_input, dtype=object)
                
                context_input = [ctx_text.encode("utf-8")]
                context_input = np.array(context_input, dtype=object)
                
                # Call rerank model with both text inputs
                result = self.models["rerank"].run(data=[query_input, context_input])
                scores.append(float(result["logits"]))
            return scores
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")