


import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import os

class TritonPythonModel:
    def initialize(self, args):
        """Initialize basic params."""
        self.model_config = json.loads(args['model_config'])
        parameters = self.model_config.get('parameters', {})
        
        self.model_version = args["model_version"]
        self.model_repository = args["model_repository"]
        self.model_path = os.path.join(self.model_repository, str(self.model_version))  # Fallback nếu cần
        
        self.max_length = int(parameters.get('max_length', {'string_value': '512'})['string_value'])
        self.logger = pb_utils.Logger
        
        self.tokenizer_cache = {} # Cache tokenizer 

    def execute(self, requests):
        responses = []
        
        for request in requests:
            try:
                model_name_or_path_tensor = pb_utils.get_input_tensor_by_name(request, "model_name_or_path")
                if model_name_or_path_tensor is None:
                    raise ValueError("model_name_or_path input is required to specify the tokenizer")
                
                model_name_or_path_np = model_name_or_path_tensor.as_numpy().flatten()
                model_name_or_path = model_name_or_path_np[0].decode('utf-8') if isinstance(model_name_or_path_np[0], bytes) else str(model_name_or_path_np[0])
                
                if model_name_or_path not in self.tokenizer_cache:
                    # self.logger.log_info(f"Loading tokenizer from: {model_name_or_path}")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained('/models/mbert.rerank.tokenizer/1')
                        self.tokenizer_cache[model_name_or_path] = tokenizer
                        # self.logger.log_info(f"Tokenizer for {model_name_or_path} loaded and cached successfully")
                    except Exception as e:
                        raise pb_utils.TritonModelException(f"Failed to load tokenizer from {model_name_or_path}: {str(e)}")
                else:
                    self.logger.log_info(f"Using cached tokenizer for: {model_name_or_path}")
                
                tokenizer = self.tokenizer_cache[model_name_or_path]  # Sử dụng tokenizer đã load
                
                query_tensor = pb_utils.get_input_tensor_by_name(request, "query")
                context_tensor = pb_utils.get_input_tensor_by_name(request, "context")
                use_for_tensor = pb_utils.get_input_tensor_by_name(request, "use_for")
                
                if use_for_tensor is None:
                    raise ValueError("use_for input is required to specify the mode")
                
                mode_np = use_for_tensor.as_numpy().flatten()
                mode = mode_np[0].decode('utf-8') if isinstance(mode_np[0], bytes) else str(mode_np[0])
                
                if mode == "query":
                    if query_tensor is None:
                        raise ValueError(f"Query input is required for {mode} mode")
                    if context_tensor is not None:
                        self.logger.log_info(f"Ignoring provided context in {mode} mode (only used for rerank)")
                    texts = [q.decode('utf-8') if isinstance(q, bytes) else str(q) for q in query_tensor.as_numpy().flatten()]
                    self.logger.log_info(f"Tokenizing {len(texts)} texts in {mode} mode")
                    encodings = tokenizer(
                        texts,
                        max_length=self.max_length,
                        padding=True,
                        truncation=True,
                        return_tensors='np'
                    )
                
                elif mode == "rerank":
                    if query_tensor is None:
                        raise ValueError("Query input is required for rerank mode")
                    if context_tensor is None:
                        raise ValueError("Context input is required for rerank mode")
                    queries_np = query_tensor.as_numpy().flatten()
                    contexts_np = context_tensor.as_numpy().flatten()
                    queries = [q.decode('utf-8') if isinstance(q, bytes) else str(q) for q in queries_np]
                    contexts = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in contexts_np]
                    
                    if len(queries) == 1:
                        query_list = [queries[0]] * len(contexts)
                    else:
                        if len(queries) != len(contexts):
                            raise ValueError("Number of queries must match number of contexts in rerank mode")
                        query_list = queries
                    
                    self.logger.log_info(f"Tokenizing {len(query_list)} query-context pairs in rerank mode")
                    encodings = tokenizer(
                        query_list,
                        contexts,
                        max_length=self.max_length,
                        padding=True,
                        truncation=True,
                        return_tensors='np'
                    )
                
                else:
                    raise ValueError(f"Invalid use_for mode: {mode}. Supported: 'query', 'rerank'")
                
                # Prepare output tensors
                batch_size = encodings['input_ids'].shape[0]
                output_tensors = [
                    pb_utils.Tensor("input_ids", encodings['input_ids'].astype(np.int64)),
                    pb_utils.Tensor("attention_mask", encodings['attention_mask'].astype(np.int64))
                ]
                if 'token_type_ids' in encodings:
                    output_tensors.append(pb_utils.Tensor("token_type_ids", encodings['token_type_ids'].astype(np.int64)))
                else:
                    token_type_ids = np.zeros((batch_size, self.max_length), dtype=np.int64)
                    output_tensors.append(pb_utils.Tensor("token_type_ids", token_type_ids))
                
                response = pb_utils.InferenceResponse(output_tensors=output_tensors)
                responses.append(response)
                
            except Exception as e:
                self.logger.log_error(f"Error processing request: {str(e)}")
                error = pb_utils.TritonError(f"Error processing request: {str(e)}")
                responses.append(pb_utils.InferenceResponse(error=error))
        
        return responses


    def finalize(self):
        """Clean up resources."""
        self.logger.log_info("Cleaning up tokenizer model")
        self.tokenizer_cache = {}
        self.logger.log_info("Tokenizer model cleaned up")
