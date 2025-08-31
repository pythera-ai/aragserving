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
query_texts = ["Con chó của bạn tên là gì ?"]
# query_bytes = [text.encode("utf-8") for text in query_texts]
context = [
    "His name is Grow",
    "Tên con mèo là Grow",
    "Bạn tên là Minh",
    "Xin chào , tôi tên là Chaos",
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

# Lấy logits từ output
logits = tokenizer_output['logits']
print("Logits:", logits)

# Tạo index sắp xếp theo thứ tự giảm dần (điểm cao nhất trước)
sorted_indices = np.argsort(logits)[::-1]  # [::-1] để đảo ngược thành giảm dần
print("Vị trí sắp xếp (cao đến thấp):", sorted_indices)

# Hoặc nếu muốn vị trí của từng phần tử trong danh sách đã sắp xếp
ranking_positions = np.argsort(np.argsort(logits)[::-1])
print("Vị trí xếp hạng của từng context:", ranking_positions)

# In kết quả chi tiết
for i, (score, context_text) in enumerate(zip(logits, context)):
    rank = ranking_positions[i] + 1  # +1 để bắt đầu từ 1 thay vì 0
    print(f"Context {i}: '{context_text}' - Score: {score:.4f} - Rank: {rank}")