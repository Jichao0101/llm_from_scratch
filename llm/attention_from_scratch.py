"""
  位置：       BOS SYS USER ASSIST A B EOS PAD
  input_ids：  1   2   3    4    5 6  7   0
  attention：  1   1   1    1    1 1  1   0
  labels：    -100 -100 -100 -100 5 6 7 -100

  约束：

  - vocab_size = 8
  - d_model = 32
  - nhead = 4
  - num_layers = 1
  - batch_first = True
  - causal mask：上三角位置为 True，表示禁止读取未来
  - padding mask：attention == 0
  - 输出 shape 应为 [1, 8, 8]
"""

import torch
from torch.nn import functional as F
import math

vocab_size = 8
d_model = 32
nhead = 4
head_dim = d_model // nhead
seq_len = 8
batch = 1

locations = [["BOS", "SYS", "USER", "ASSISTANT", "A", "B", "EOS", "PAD"]] #每个位置的输入
input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]]) #每个位置的id
attention = torch.ones(batch, seq_len)
attention[0, -1] = 0  #每个位置的attention,即标签是否参与训练
labels = torch.tensor([[-100, -100, -100, -100, 5, 6, 7, -100]]) #每个位置的标签
shift_labels = labels[: , 1:].contiguous() #每个位置预测的标签

vocab_matrix = torch.randn(vocab_size, d_model) # 词汇表
token_embedding = vocab_matrix[input_ids]  # 每个位置的词向量
position_matrix = torch.randn(seq_len, d_model)
position_ids = torch.arange(seq_len).unsqueeze(0)
position_embedding = position_matrix[position_ids]
embedding = token_embedding + position_embedding # 每个位置的输入向量 [1, 8, 32]
casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) # 对角线之上的三角位置为 True，表示禁止读取未来 [8,8]
padding_mask = (attention == 0) # 不允许读取 PAD [1, 8]
padding_mask = padding_mask[:, None, None, :] #[1, 1, 1, 8]

W_k = torch.randn(d_model, d_model) # 权重矩阵不会有batch维度
W_v = torch.randn(d_model, d_model)
W_q = torch.randn(d_model, d_model)
W_o = torch.randn(d_model, d_model)
lm_head = torch.randn(d_model,vocab_size)


q = torch.matmul(embedding, W_q) # shape [1, 8, 32] 
q = q.view(batch, seq_len, nhead, head_dim).transpose(1, 2) # [1. 4. 8. 8]
k = torch.matmul(embedding, W_k)
k = k.view(batch, seq_len, nhead, head_dim).transpose(1, 2)
v = torch.matmul(embedding, W_v)
v= v.view(batch, seq_len, nhead, head_dim).transpose(1, 2)

attn_weights = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(head_dim) # shape [1, 4, 8, 8]
attn_weights = attn_weights.masked_fill(casual_mask.bool(), float("-inf"))
attn_weights = attn_weights.masked_fill(padding_mask.bool(), float("-inf"))
attn_probs = torch.softmax(attn_weights, dim = -1)
hidden_state = torch.matmul(attn_probs, v) # [1, 4, 8, 8]
hidden_state = hidden_state.transpose(1, 2).contiguous().view(batch, seq_len, d_model)  # [1, 8, 32]
output = torch.matmul(hidden_state, W_o) + embedding  # [1, 8, 32]
logits = torch.matmul(output, lm_head) # [1, 8, 8]
print(logits.shape)
shift_logits = logits[:, :-1,:].contiguous() # [1, 7, 8]
print(shift_logits.shape)
loss = F.cross_entropy(
    shift_logits.view(-1, vocab_size),
    shift_labels.view(-1),
    ignore_index = -100
)
print(loss)