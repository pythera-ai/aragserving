from trism import TritonModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = TritonModel(

    model= 'mbert-retrieve-ctx',
    version= 1,
    url= 'localhost:7001',
    grpc= True
)


#init tokenizer name
model_name = 'pythera/mbert-retrieve-ctx-base'
model_name = np.array([model_name], dtype=object)


# create query and context 
# query_texts = ["What is the capital of France?"]
# query_bytes = [text.encode("utf-8") for text in query_texts]
context_texts = [
    "His name is Grow.",
    "Tên con chó là .",
    "Thủ đô của pháp là Việt Nam"
] 

input_merge = ['Con chó tên là gì?']
# input_merge.append(query_texts)
input_merge.append(context_texts)

input_merge = [str(input_merge)]
input_merge = [text.encode("utf-8") for text in input_merge]

input_merge = np.expand_dims(input_merge, axis=0)  

input_merge = np.array(input_merge, dtype=object)
print(input_merge.shape)


data_input = [input_merge, model_name]

tokenizer_output = model.run(data=data_input)
# print("Output for  model:", tokenizer_output.shape)
