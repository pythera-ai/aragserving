
import hashlib
from io import BytesIO
import numpy as np
import docx
from pathlib import Path
from fastapi import HTTPException

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

def prepare_text_input(text: str) -> np.ndarray:
    """Prepare text input for Triton model"""
    input_text = [text.encode("utf-8")]
    input_array = np.array(input_text, dtype=object)
    return np.expand_dims(input_array, axis=0)