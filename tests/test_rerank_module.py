from trism import TritonModel
import numpy as np

model = TritonModel(

    model= 'mbert.rerank',
    version= 1,
    url= 'localhost:7001',
    grpc= True
)


#init tokenizer name
model_name = 'pythera/mbert-retrieve-ctx-base'
model_name = np.array([model_name], dtype=object)


# create query and context 
query_texts = ["Vì sao khi bị sốt, người bệnh hay cảm thấy lạnh?"]
# query_bytes = [text.encode("utf-8") for text in query_texts]
context = [
    "Ở người đang bị sốt, việc đắp chăn sẽ không giúp xua tan cơn lạnh mà càng khiến cơ thể khó thoát nhiệt, dẫn đến tình trạng sốt kéo dài",
    "Tôi bị ngu",
    "Tôi khôg có thông tin",
    "Tôi không biết gì về cảm lạnh cả ",
]

input_merge = []
input_merge.append(query_texts)
input_merge.append(context)

input_merge = [str(input_merge)]
input_merge = [text.encode("utf-8") for text in input_merge]

input_merge = np.array(input_merge, dtype=object)

data_input = [input_merge, model_name]

tokenizer_output = model.run(data=data_input)
print("Output for  model:", tokenizer_output)
