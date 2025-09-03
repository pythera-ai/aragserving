
import hashlib
from io import BytesIO
import numpy as np
import docx
from pathlib import Path
from fastapi import HTTPException
from typing import List, Union , Dict

####################
# Utility Functions
####################

def generate_md5_hash(text: str) -> str:
    """Generate MD5 hash for text chunk"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text content from uploaded file"""
    file_ext = Path(filename).suffix.lower()
    
    if file_ext == '.txt' or file_ext == '.md':
        return file_content.decode('utf-8')
    elif file_ext == '.docx':
        doc = docx.Document(BytesIO(file_content))
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {file_ext}. Supported formats: .txt, .md, .docx"
        )

def prepare_input_format(list_text:List,  expand_dims:bool = False) -> np.array:
    """Prepare input format for Triton model"""

    # Đảm bảo tất cả text đều là string trước khi encode
    object_text = []
    for text in list_text:
        if isinstance(text, str):
            object_text.append(text.encode("utf-8"))
        else:
            # Convert to string nếu không phải string
            object_text.append(str(text).encode("utf-8"))
    
    input_array = np.array(object_text, dtype=object)

    if expand_dims:
        input_array = np.expand_dims(input_array, axis=0)
    return input_array

def prepare_rerank_input(query_texts: List[str], context_texts: List[str]) -> List[str]:
    """Prepare input format for Triton model"""
    input_merge = []
    input_merge.append(query_texts)
    input_merge.append(context_texts)
    input_merge = [str(input_merge)]
    input_merge = [text.encode("utf-8") for text in input_merge]
    input_merge_array = np.array(input_merge, dtype = object)

    return input_merge_array


def prepare_input_for_tokenizer(tokenizer_name:str) -> np.array:
    """Prepare input format for Triton model"""
    input_tokenizer_array = np.array([tokenizer_name], dtype=object)
    
    return input_tokenizer_array

def list_to_array(list_input:list) -> np.array:
    """Convert list to numpy array"""
    return np.array(list_input)


def extract_tokenizer_name(tokenizer:Dict, model_name:str) -> str:
    """Extract tokenizer name from model config"""
    try:

        return tokenizer[model_name]['name']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract tokenizer name: {str(e)}")

