'''
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

  nn.Embedding
  nn.TransformerEncoderLayer
  nn.LayerNorm
  nn.Linear
  F.cross_entropy
  torch.optim.AdamW

  请把职责分成两组：

  1. TinyCausalLM.forward() 接收什么、执行什么、返回什么？
  2. train_step() 接收什么、执行什么、记录什么？

  要求覆盖 causal mask、padding mask、label shift、backward、梯度范数和参数变化证据
'''

import torch
from torch import nn as nn
from torch.nn import functional as F

vocab_size = 8
d_model = 32
nhead = 4
num_layers = 1

transformer_layer = nn.TransformerEncoderLayer(
	d_model=d_model,
	nhead= nhead, 
	dim_feedforward=2048, 
	dropout=0.1, 
	activation="relu", 
	layer_norm_eps=0.00001, 
	batch_first=True,
	norm_first=True,
	bias=True,
	)

class TinyCausalLM(nn.Module):

	def __init__(self):
		super().__init__()
		self.vocab = nn.Embedding(vocab_size, d_model)
		self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers, enable_nested_tensor=False)
		self.layer_norm = nn.LayerNorm(d_model)
		self.lm_head = nn.Linear(d_model, vocab_size)


	def forward(self, input_ids,  attention):
		seq_len = input_ids.shape[1]
		causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
		input_embedding = self.vocab(input_ids)
		padding_mask = attention == 0
		hidden_state = self.transformer(input_embedding, mask=causal_mask, src_key_padding_mask=padding_mask)
		hidden_state = self.layer_norm(hidden_state)
		logits = self.lm_head(hidden_state)
		return logits

model = TinyCausalLM()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
	
locations = ["BOS", "SYS", "USER", "ASSIST", "A", "B", "EOS", "PAD"]
input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]]) # [1, 8]
attention = torch.tensor([[1, 1, 1, 1, 1, 1 ,1, 0]]) # [1, 8]
labels = torch.tensor([[-100, -100, -100, -100, 5, 6, 7, -100]])
padding_mask = (attention == 0)

def train_step(
		model,
		optimizer,
		input_ids,
		labels,
		attention_mask
):
      
	model.train()
	old_weight = model.lm_head.weight.detach().clone()

	optimizer.zero_grad()
	logits = model(input_ids, attention_mask)
	shift_logits = logits[:, :-1, :]
	shift_labels = labels[:, 1:]
	loss =F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100)
	loss.backward()


	total_norm = 0.0

	for p in model.parameters():
		if p.grad is not None:
			param_norm = p.grad.data.norm(2)
			total_norm += (
                param_norm.item()
                ** 2
            )
	grad_norm = total_norm ** 0.5

	optimizer.step()

    # 参数变化证据
	weight_delta = (model.lm_head.weight.detach() - old_weight).abs().mean()
	metrics = {"loss":loss.item(), "grad_norm":grad_norm, "lm_head_delta":weight_delta.item()}
	return metrics

metrics = train_step(model, optimizer=optimizer, input_ids=input_ids, labels=labels, attention_mask=attention)

'''
 evaluate_single_batch()，要求：

  - 不接收 optimizer；
  - 保存并恢复原来的 train/eval 状态；
  - 使用 torch.no_grad()；
  - 正确执行 label shift 和 ignore_index=-100；
  - 返回 eval_loss、correct_tokens、valid_tokens、token_accuracy。
'''

def evaluate_single_batch(
		model,
		input_ids,
		labels,
		attention_mask
):
	model.eval()
	with torch.no_grad():
		logits = model(input_ids, attention_mask)
		shift_logits = logits[:, :-1, :]
		shift_labels = labels[:, 1:]
		loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100, reduce="sum")
		cls = torch.argmax(shift_logits, dim=2)
		print(cls)
		correct_tokens = torch.sum(cls == shift_labels)
		valid_tokens = torch.sum(shift_labels != -100)
		token_accuray = correct_tokens / valid_tokens
		return loss, correct_tokens, valid_tokens, token_accuray


loss, correct_tokens, valid_tokens, token_accuracy = evaluate_single_batch(model, input_ids=input_ids, labels=labels, attention_mask=attention)





