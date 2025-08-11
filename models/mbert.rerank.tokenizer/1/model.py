import json
import numpy as np
import os 
import triton_python_backend_utils as pb_utils
from huggingface_hub import login, snapshot_download

from typing import List , Dict , Any
import logging
from transformers import AutoTokenizer
import ast
import dotenv
dotenv.load_dotenv()
# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
HF_TOKEN = os.getenv('HF_TOKEN')


class TritonPythonModel:
    
    """
    This pipeline is use to call the tokenizer model and return the input_ids, attention_mask, token_type_ids
    """

    def initialize(self, args: Dict[str, Any]) -> None:

        self.model_config = args['model_config']
        self.model_version = args['model_version']
        self.model_path = args['model_repository'] + '/' + args['model_version']

    def execute(self, requests):
        """Process inference requests and return tokenized outputs."""
        responses = []
        
        for request in requests:
            try:

                # Get input text)
                input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
                input_texts = input_tensor.as_numpy()  # Shape: [batch_size, 1]

                # Flatten and decode the input texts
                input_texts = [text.decode('utf-8') for text in input_texts.flatten()]
                
                query_list, context_list = self.split_query_context(input_texts)
              
                # Get tokenizer name 
                tokenizer_name_tensor = pb_utils.get_input_tensor_by_name(request, 'tokenizer_name')

                tokenizer_name = tokenizer_name_tensor.as_numpy().flatten()[0].decode('utf-8')
                
                tokenizer_save_dir = os.path.join(self.model_path, tokenizer_name)
                
                if os.path.exists(tokenizer_save_dir):
                    logger.info(f"Tokenizer already downloaded: {tokenizer_name}")
                else:
                    self.download_tokenizer(tokenizer_name,tokenizer_save_dir)

                # Load tokenizer from model_path
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_dir)
                    logger.info(f"Successfully loaded tokenizer: {tokenizer_name}")
                except:
                    logger.info(f"Failed to load tokenizer: {tokenizer_name}")
                
                # Tokenize the input texts - NO TRUNCATION for long documents
                
                encodings = tokenizer(
                    query_list,
                    context_list,
                    # max_length=None,  # No max length limit
                    padding=True,    # No padding for variable length
                    truncation=True, # No truncation
                    return_tensors='np' # Return NumPy arrays
                    # add_special_tokens=False
                )

                # Create dummy input_ids and attention_mask
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

                
                # Ensure proper shape - add batch dimension if needed

            except Exception as e:
                error = pb_utils.TritonError(f"Error processing request: {str(e)}")
                responses.append(pb_utils.InferenceResponse(error=error))
        
        return responses
    
    def split_query_context(self, input_texts):

        # convert string to list 
        list_combine= ast.literal_eval(input_texts[0])    

        # get context
        context_list = list_combine[1]

        # get query and repeat it for each context
        query_list = list_combine[0] * len(context_list)


        return query_list, context_list
    
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

    def finalize(self):
        """Clean up resources."""
        logger.info("Cleaning up ")