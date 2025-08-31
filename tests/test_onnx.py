from trism import TritonModel
import numpy as np

model = TritonModel(
    model='qwen3-retrieve-sym.model',
    version=1,
    url='localhost:7001',
    grpc=True
)

# Create test input data based on model config
batch_size = 1
seq_length = 512

# input_ids: [batch_size, seq_length] - INT64
input_ids = np.random.randint(0, 30000, (batch_size, seq_length), dtype=np.int64)

# attention_mask: [batch_size, seq_length] - INT64  
attention_mask = np.ones((batch_size, seq_length), dtype=np.int64)

# Create data input format with proper tensor names
data_input = [
    input_ids,
    attention_mask
]

try:
    onnx_output = model.run(data=data_input)
    print("ONNX model output keys:", list(onnx_output.keys()) if onnx_output else "No output")
    if onnx_output:
        for key, value in onnx_output.items():
            print(f"{key}: shape={value.shape if hasattr(value, 'shape') else type(value)}, dtype={value.dtype if hasattr(value, 'dtype') else type(value)}")
            if hasattr(value, 'shape'):
                print(f"  Sample values (first 5): {value.flatten()[:5]}")
    print("ONNX model test completed successfully!")
except Exception as e:
    print(f"Error testing ONNX model: {e}")