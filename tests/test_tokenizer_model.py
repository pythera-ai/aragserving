from trism import TritonModel
import numpy as np

model = TritonModel(
    model='qwen3-retrieve-sym',
    version=1,
    url='localhost:7001',
    grpc=True
)

# Init tokenizer name
model_name = 'Qwen/Qwen3-Embedding-0.6B'
model_name_array = np.array([model_name], dtype=object)

# Use string format instead of bytes
query_texts = ["Hey who are you ?"]
query_numpy = np.array(query_texts, dtype=object)

# Create max_length parameter
max_length = np.array([512], dtype=np.int32)

# Correct data input format with proper tensor names
data_input = [
    query_numpy, model_name_array
]

try:
    tokenizer_output = model.run(data=data_input)
    print("Tokenizer output keys:", list(tokenizer_output.keys()) if tokenizer_output else "No output")
    if tokenizer_output:
        for key, value in tokenizer_output.items():
            print(f"{key}: shape={value.shape if hasattr(value, 'shape') else type(value)}")
except Exception as e:
    print(f"Error: {e}")
