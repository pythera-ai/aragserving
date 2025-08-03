import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import os
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    def initialize(self, args):
        """Initialize the tokenizer based on the config parameters."""
        self.model_config = json.loads(args['model_config'])
        # Load parameters from config
        parameters = self.model_config.get('parameters', {})
        self.max_length = int(parameters.get('max_length', {'string_value': '128'})['string_value'])
        self.logger = logger
        
        try:
            model_name_or_path = args["model_repository"].split('.tokenizer')[0] + '.model'
            model_name_or_path = os.path.join(model_name_or_path, args["model_version"])
            # load model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.logger.info("Tokenizer loaded successfully")
        except Exception as e:
            raise pb_utils.TritonModelException(f"Failed to load tokenizer: {str(e)}")


    def execute(self, requests):
        """Process inference requests and return tokenized outputs."""
        responses = []
        
        for request in requests:
            try:
                # Get input tensor (text as string)
                input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
                input_texts = input_tensor.as_numpy()  # Shape: [batch_size, 1]
            
                # Flatten and decode the input texts
                input_texts = [text.decode('utf-8') for text in input_texts.flatten()]
                
                # Tokenize the input texts - NO TRUNCATION for long documents
                encodings = self.tokenizer(
                    input_texts,
                    padding=False,    # No padding for variable length
                    truncation=False, # No truncation
                    return_tensors='np',  # Return NumPy arrays
                    add_special_tokens=False
                )
                
                # Ensure proper shape - add batch dimension if needed
                input_ids = encodings['input_ids']
                attention_mask = encodings['attention_mask']
                
                if input_ids.ndim == 1:
                    input_ids = input_ids.reshape(1, -1)
                if attention_mask.ndim == 1:
                    attention_mask = attention_mask.reshape(1, -1)
                
                self.logger.info(f"Tokenized sequence length: {input_ids.shape[1]} tokens")
                
                # Prepare output tensors
                output_tensors = [
                    pb_utils.Tensor("input_ids", input_ids.astype(np.int64)),
                    pb_utils.Tensor("attention_mask", attention_mask.astype(np.int64))
                ]
                
                # Include token_type_ids if available (e.g., for BERT)
                if 'token_type_ids' in encodings:
                    output_tensors.append(pb_utils.Tensor("token_type_ids", encodings['token_type_ids'].astype(np.int64)))
                
                # Create response
                response = pb_utils.InferenceResponse(output_tensors=output_tensors)
                responses.append(response)
                
            except Exception as e:
                error = pb_utils.TritonError(f"Error processing request: {str(e)}")
                responses.append(pb_utils.InferenceResponse(error=error))
        
        return responses

    def finalize(self):
        """Clean up resources."""
        self.logger.info("Cleaning up tokenizer model")
        self.tokenizer = None
        self.logger.info("Tokenizer model cleaned up")
