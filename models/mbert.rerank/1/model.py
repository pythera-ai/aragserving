
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

    def _call_tokenizers(self, text: np.ndarray, tokenizer_name: np.ndarray):
        """
        Calls the tokenizer model to get input_ids, attention_mask, and token_type_ids.
        Assumes one query repeated for multiple contexts.
        """
   
        # Create Triton tensors (note: strings are handled as object dtype with bytes)
        inputs = [pb_utils.Tensor("text", text), pb_utils.Tensor("tokenizer_name", tokenizer_name)]

        # Requested outputs
        requested_outputs = ["input_ids", "attention_mask", "token_type_ids"]

        # Create and execute inference request to tokenizer model
        infer_request = pb_utils.InferenceRequest(
            model_name="mbert.rerank.tokenizer",
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
            model_name="mbert.rerank.model",
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
                text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
                tokenizer_name_tensor = pb_utils.get_input_tensor_by_name(request, "tokenizer_name")

                logger.info(f'Input text_tensor tpye: {type(text_tensor)}')
                logger.info(f'Input tokenizer_name_tensor tpye: {type(tokenizer_name_tensor)}')


                # Step 1: Call tokenizer via helper function
                input_ids, attention_mask, token_type_ids = self._call_tokenizers(text_tensor.as_numpy(), tokenizer_name_tensor.as_numpy() )

                # Step 2: Call platform model via helper function
                logits = self._call_model_platform(input_ids, attention_mask, token_type_ids)

                # For simplicity, flatten logits to 1D array of scores
                scores = logits.flatten().astype(np.float32)
                
                # Invert scores: lower scores become higher (more relevant)
                # This is because the model might be scoring negative examples higher
                scores = -scores
                
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
