import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import os
import logging
from huggingface_hub import login, snapshot_download

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import dotenv
dotenv.load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the tokenizer based on the config parameters."""
        self.model_config = json.loads(args['model_config'])
        self.model_version = args['model_version']
        self.model_path = args['model_repository'] + '/' + args['model_version']
        # Load parameters from config
        self.logger = pb_utils.Logger
        

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
                
                  # Get tokenizer name from input

                tokenizer_name_tensor = pb_utils.get_input_tensor_by_name(request, 'tokenizer_name')
                tokenizer_name = tokenizer_name_tensor.as_numpy().flatten()[0].decode('utf-8')

                # Save tokenizer to save dir 
                tokenizer_save_dir = os.path.join(self.model_path, tokenizer_name)

                if os.path.exists(tokenizer_save_dir):
                    logger.info(f"Tokenizer already downloaded: {tokenizer_name}")
                else:
                    self.download_tokenizer(tokenizer_name,tokenizer_save_dir)
                logger.info(f'Tokenizer saved to {tokenizer_save_dir}')
                # Load tokenizer from model_path
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_dir)
                    logger.info(f"Successfully loaded tokenizer: {tokenizer_name}")
                except: 
                    logger.info(f"Failed to load tokenizer: {tokenizer_name}")

                try: # Tokenize the input texts
                    encodings = tokenizer(
                        input_texts,
                        padding='max_length',
                        truncation=True,
                        return_tensors='np'  # Return NumPy arrays
                    )
                except Exception as e:
                    error = pb_utils.TritonError(f"Error processing request: {str(e)}")
                    responses.append(pb_utils.InferenceResponse(error=error))
                    return responses

                
                # Prepare output tensors
                output_tensors = [
                    pb_utils.Tensor("input_ids", encodings['input_ids'].astype(np.int64)),
                    pb_utils.Tensor("attention_mask", encodings['attention_mask'].astype(np.int64))
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
        self.logger.log_info("Cleaning up tokenizer model")
        self.tokenizer = None
        self.logger.log_info("Tokenizer model cleaned up")


    def download_tokenizer(self,tokenizer_name:str,save_dir:str):

                """
                Download tokenizer from huggingface hub

                Args:
                    tokenizer_name (str): name of the tokenizer
                    save_dir (str): directory to save the tokenizer
                    token (str): huggingface token
                """

                snapshot_download(repo_id=tokenizer_name, local_dir=save_dir, token= HF_TOKEN,
                                allow_patterns=["config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"],
                            )
                