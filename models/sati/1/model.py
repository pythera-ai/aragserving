import triton_python_backend_utils as pb_utils
import numpy as np
from typing import List
from transformers import AutoTokenizer
import os
import logging
# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:

    def initialize(self, args):
        """Initialize the tokenizer based on the config parameters."""
        self.bls_repository = args['model_repository']
        self.model_version = args['model_version']
        self.tokenizer_repository = self.bls_repository.split("/")[-1] + ".tokenizer"
        self.model_repository = self.bls_repository.split("/")[-1] + ".model"
        # Get tokenizer folders path
        tokenizer_folder_path = self.bls_repository + ".tokenizer"
        self.tokenizer_folder_path = os.path.join(tokenizer_folder_path, self.model_version)
        self.logger = logging.getLogger(__name__)

    def _call_tokenizers(self, text: np.ndarray, tokenizer_name: np.ndarray):
            """
            Calls the tokenizer model to get input_ids, attention_mask, and token_type_ids.
            Assumes one query repeated for multiple contexts.
            """
            text = np.array([text.encode("utf-8")],  dtype=object)
            tokenizer_name = tokenizer_name.as_numpy()


            # Create Triton tensors (note: strings are handled as object dtype with bytes)
            inputs = [pb_utils.Tensor("text", text), pb_utils.Tensor("tokenizer_name", tokenizer_name)]

            # Requested outputs
            requested_outputs = ["input_ids", "attention_mask", "token_type_ids"]

            # Create and execute inference request to tokenizer model
            infer_request = pb_utils.InferenceRequest(
                model_name="sati.tokenizer",
                requested_output_names=requested_outputs,
                inputs=inputs
            )
            infer_response = infer_request.exec()

            if infer_response.has_error():
                # raise RuntimeError(f"Tokenizer inference failed: {infer_response.error().message()}")
                self.logger.error(f"Tokenizer inference failed: {infer_response.error().message()}")
            # Extract outputs
            input_ids = pb_utils.get_output_tensor_by_name(infer_response, "input_ids").as_numpy()
            attention_mask = pb_utils.get_output_tensor_by_name(infer_response, "attention_mask").as_numpy()

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }



    def execute(self, requests):

        responses = []
        for request in requests:
            # Input text tensor
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            text_bytes_array = text_tensor.as_numpy()
            # Handle both [1,1] and [1] shapes
            if text_bytes_array.ndim == 2:
                text_bytes = text_bytes_array[0, 0]
            else:
                text_bytes = text_bytes_array[0]
            text = text_bytes.decode("utf-8")
            self.logger.info(f"Processing text length: {len(text)} characters")

            # Get max_tokens parameter (optional, default to 256)
            max_tokens = 256  # Default value
            max_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "max_tokens")
            if max_tokens_tensor is not None:
                max_tokens_array = max_tokens_tensor.as_numpy()
                # Handle different shapes
                if max_tokens_array.ndim == 2:
                    max_tokens = int(max_tokens_array[0, 0])
                else:
                    max_tokens = int(max_tokens_array[0])
                self.logger.info(f"Using max_tokens: {max_tokens}")
            else:
                self.logger.info(f"Using default max_tokens: {max_tokens}")


            # Get tokenizer_name
            tokenizer_name_tensor = pb_utils.get_input_tensor_by_name(request, "tokenizer_name")

            # Call tokenizer
            encoded = self._call_tokenizers(text, tokenizer_name=tokenizer_name_tensor )
                
            # get input_ids and log total tokens
            input_ids = encoded["input_ids"][0]  # 1-D array của token IDs
            total_len = len(input_ids)
            self.logger.info(f"Total tokens after tokenization: {total_len}")
            
            # Chunks of max length 512 tokens - logic đơn giản như code 2
            text_responses = []
            start = 0
            times = 0

            while start + 512 < total_len:
                input_model = {}
                for name, data in encoded.items():
                    # Slice input tokens for this chunk
                    arr = data[:, start:start+512]
                    # input_ids should be int64, attention_mask should be fp16
                    if name == "attention_mask":
                        input_model[name] = arr.astype(np.float16)
                    else:
                        input_model[name] = arr.astype(np.int64)
                
                platform_request = pb_utils.InferenceRequest(
                    model_name=self.model_repository,
                    requested_output_names=["logits"],
                    inputs=[pb_utils.Tensor(name, data) for name, data in input_model.items()]
                )
                platform_response = platform_request.exec()
                if platform_response.has_error():
                    self.logger.error(f"Platform model error: {platform_response.error().message()}")
                    start += 512
                    times += 1
                    continue
                    
                logits_tensor = pb_utils.get_output_tensor_by_name(platform_response, "logits")
                if logits_tensor is None:
                    self.logger.error("No logits output from platform model")
                    start += 512
                    times += 1
                    continue
                    
                logits = logits_tensor.as_numpy()[0]
                
                # Collect indices where output > 0 
                all_index = [idx + 1 for idx, val in enumerate(logits) if val > 0]
                all_index = [512 * times + idx for idx in all_index]
                
                if all_index:
                    text_responses.extend(all_index)
                    start = text_responses[-1]  
                else:
                    start += 512  # No boundaries found, move to next chunk
                times += 1

            # Process final segment if remaining 
            if start < total_len:
                input_model = {}
                for name, data in encoded.items():
                    arr = data[:, start:]
                    # input_ids should be int64, attention_mask should be fp16
                    if name == "attention_mask":
                        input_model[name] = arr.astype(np.float16)
                    else:
                        input_model[name] = arr.astype(np.int64)
                
                platform_request = pb_utils.InferenceRequest(
                    model_name=self.model_repository,
                    requested_output_names=["logits"],
                    inputs=[pb_utils.Tensor(name, data) for name, data in input_model.items()]
                )
                platform_response = platform_request.exec()
                if not platform_response.has_error():
                    logits_tensor = pb_utils.get_output_tensor_by_name(platform_response, "logits")
                    if logits_tensor is not None:
                        logits = logits_tensor.as_numpy()[0]
                        all_index = [idx + 1 for idx, val in enumerate(logits) if val > 0]
                        all_index = [start + idx for idx in all_index]
                        text_responses.extend(all_index)

            # Decode the chunks from token indices
            chunks = []
            prev = 0
            
            self.logger.info(f"Found {len(text_responses)} sentence boundaries")

            # Load tokenizer from model_path

            try:
                tokenizer_name = tokenizer_name_tensor.as_numpy().flatten()[0].decode('utf-8')
                tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.tokenizer_folder_path, tokenizer_name))   
                logger.info(f"Successfully loaded tokenizer for decode: {tokenizer_name}")
            except:
                self.logger.info(f'Can not load tokenizer from {tokenizer_name}')


            if not text_responses:
                # No boundaries found, decode entire text
                tokens = encoded["input_ids"][0][:]
                chunks.append(tokenizer.decode(tokens.tolist()))
                self.logger.info(f"No boundaries found, returning full text as single chunk")
            else:
                # Decode chunks based on found boundaries 
                for idx in text_responses:
                    if idx > prev:  # Thêm check này để tránh duplicate
                        tokens = input_ids[prev:idx]
                        decoded_chunk = tokenizer.decode(tokens.tolist())
                        chunks.append(decoded_chunk)
                        prev = idx
                
                # Don't forget the tail chunk 
                if prev < total_len:
                    tail_tokens = input_ids[prev:]
                    tail_str = tokenizer.decode(tail_tokens.tolist())
                    if tail_str:
                        chunks.append(tail_str)
            
            chunks = [c for c in chunks if c.strip()]
            self.logger.info(f"After decoding: {len(chunks)} chunks")
           
            
            # Apply merge with max_tokens of model's max_length
            chunks = self.merge_subtexts_fix(tokenizer, chunks, max_tokens=max_tokens)

            # Create output tensor of strings
            out_bytes = [c.encode("utf-8") for c in chunks]
            out_array = np.array(out_bytes, dtype=np.bytes_)
            out_tensor = pb_utils.Tensor("logits", out_array)

            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)

        return responses

    def merge_subtexts_fix(self,tokenizer, list_sub_text, max_tokens=256):  # Đổi default về 256
        """Merge function from reference model"""
        merged_texts = []
        current_num_token = 0
        current_merge = ""

        for subtext in list_sub_text:
            num_tokens = len(tokenizer(subtext)["input_ids"])
            if current_num_token + num_tokens > max_tokens:
                if current_merge:
                    merged_texts.append(current_merge.strip())
                current_merge = subtext
                current_num_token = num_tokens
            else:
                current_merge += " " + subtext
                current_num_token += num_tokens

        if current_merge:
            merged_texts.append(current_merge.strip())
        return merged_texts
    
    
    def finalize(self):
        print("Finalizing...")

