from trism import TritonModel
import numpy as np

model = TritonModel(

    model= 'gemma-300m-embedding-sym',
    version= 1,
    url= 'localhost:7001',
    grpc= True
)


#init tokenizer name
model_name = 'onnx-community/embeddinggemma-300m-ONNX'
model_name = np.array([model_name], dtype=object)

context_texts = [
    "Paris is the capital of France."
] 


query = [text.encode("utf-8") for text in context_texts]
# input_merge = context_texts[0].encode("utf-8") 
query_numpy = np.array(query)  


data_input = [query_numpy, model_name]

tokenizer_output = model.run(data=data_input)
print("Output for  model:", tokenizer_output)
