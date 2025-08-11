from transformers import AutoTokenizer
import os

def test_load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("/home/chaos/Documents/chaos/repo/aragserving/models/sati.tokenizer/1/pythera/sat",)
    assert tokenizer is not None
test_load_tokenizer()