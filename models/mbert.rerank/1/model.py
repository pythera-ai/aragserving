import json
import numpy as np
import triton_python_backend_utils as pb_utils
from typing import Dict, Any
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import json
import numpy as np
import triton_python_backend_utils as pb_utils
from typing import Dict, Any
import logging
from transformers import AutoTokenizer
# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """
    Pipeline model that connects tokenizer and mbert reranking model.
    NOTE: This is a simplified version that works within single Triton instance.
    For true microservice architecture, use separate containers with HTTP clients.
    """

    def initialize(self, args: Dict[str, Any]) -> None:
        self.model_config = json.loads(args['model_config'])
        # Hardcode tokenizer parameters here; in a real setup, load from config parameters
        self.tokenizer_model_name = AutoTokenizer.from_pretrained('/models/mbert.rerank.tokenizer/1')
        self.use_for = "rerank"  # Example use; adjust based on tokenizer logic
        logger.info("Pipeline model initialized")

    def _call_tokenizers(self, queries: np.ndarray, contexts: np.ndarray):
        """
        Calls the tokenizer model to get input_ids, attention_mask, and token_type_ids.
        Assumes one query repeated for multiple contexts.
        """
        # Prepare inputs for tokenizer
        num_contexts = len(contexts)
        # Repeat query for each context to form pairs
        repeated_queries = np.repeat(queries, num_contexts).astype(object)
        model_name_arr = np.array([self.tokenizer_model_name], dtype=object)
        use_for_arr = np.array([self.use_for], dtype=object)

        # Create Triton tensors (note: strings are handled as object dtype with bytes)
        inputs = [
            pb_utils.Tensor("model_name_or_path", model_name_arr),
            pb_utils.Tensor("query", repeated_queries),
            pb_utils.Tensor("context", contexts),
            pb_utils.Tensor("use_for", use_for_arr)
        ]

        # Requested outputs
        requested_outputs = ["input_ids", "attention_mask", "token_type_ids"]

        # Create and execute inference request to tokenizer model
        infer_request = pb_utils.InferenceRequest(
            model_name="tokenizer",
            requested_output_names=requested_outputs,
            inputs=inputs
        )
        infer_response = infer_request.exec()

        if infer_response.has_error():
            raise pb_utils.TritonError(f"Tokenizer inference failed: {infer_response.error().message()}")

        # Extract outputs
        input_ids = pb_utils.get_output_tensor_by_name(infer_response, "input_ids").as_numpy()
        attention_mask = pb_utils.get_output_tensor_by_name(infer_response, "attention_mask").as_numpy()
        token_type_ids = pb_utils.get_output_tensor_by_name(infer_response, "token_type_ids").as_numpy()

        return input_ids, attention_mask, token_type_ids

    def _call_model_platform(self, input_ids: np.ndarray, attention_mask: np.ndarray, token_type_ids: np.ndarray):
        """
        Calls the platform (ONNX) model to get logits scores.
        Assumes input shapes are compatible (e.g., batch_size <= 10).
        """
        # Create Triton tensors for platform model
        inputs = [
            pb_utils.Tensor("input_ids", input_ids),
            pb_utils.Tensor("attention_mask", attention_mask),
            pb_utils.Tensor("token_type_ids", token_type_ids)
        ]

        # Requested outputs
        requested_outputs = ["logits"]

        # Create and execute inference request to platform model
        infer_request = pb_utils.InferenceRequest(
            model_name="platform",
            requested_output_names=requested_outputs,
            inputs=inputs
        )
        infer_response = infer_request.exec()

        if infer_response.has_error():
            raise pb_utils.TritonError(f"Platform inference failed: {infer_response.error().message()}")

        # Extract logits
        logits = pb_utils.get_output_tensor_by_name(infer_response, "logits").as_numpy()

        return logits

    def execute(self, requests):
        logger.info(f">>> Got {len(requests)} requests")
        responses = []

        for request in requests:
            try:
                # Log input info
                input_names = [inp.name() for inp in request.inputs()]
                input_shapes = [inp.shape() for inp in request.inputs()]
                logger.info(f">>> Input names: {input_names}")
                logger.info(f">>> Input shapes: {input_shapes}")

                # Get input tensors
                query_tensor = pb_utils.get_input_tensor_by_name(request, "query")
                context_tensor = pb_utils.get_input_tensor_by_name(request, "context")

                if query_tensor is None or context_tensor is None:
                    raise RuntimeError("Missing required input tensors: query and/or context")

                # Convert to numpy (flatten for simplicity, assuming single query)
                queries_np = query_tensor.as_numpy().flatten()
                contexts_np = context_tensor.as_numpy().flatten()

                # Decode strings
                queries = [q.decode('utf-8') if isinstance(q, bytes) else str(q) for q in queries_np]
                contexts = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in contexts_np]

                if len(queries) != 1:
                    raise ValueError("Pipeline expects a single query per request")

                logger.info(f"Processing query: {queries[0]} with {len(contexts)} contexts")

                # Step 1: Call tokenizer via helper function
                input_ids, attention_mask, token_type_ids = self._call_tokenizers(np.array(queries, dtype=object), np.array(contexts, dtype=object))

                # Step 2: Call platform model via helper function
                logits = self._call_model_platform(input_ids, attention_mask, token_type_ids)

                # For simplicity, flatten logits to 1D array of scores
                scores = logits.flatten().astype(np.float32)
                logger.info(f"Generated scores: {scores}")

                # Create response
                output_tensor = pb_utils.Tensor("logits", scores)
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(response)

            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                error = pb_utils.TritonError(f"Pipeline error: {str(e)}")
                responses.append(pb_utils.InferenceResponse(error=error))

        return responses


    def finalize(self):
        logger.info("Cleaning up pipeline model...")
        logger.info("Pipeline model cleaned up")
