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
import copy

torch.manual_seed(42)

vocab_size = 8
d_model = 32
nhead = 4
num_layers = 1


class TinyCausalLM(nn.Module):
	def __init__(self):
		super().__init__()
		self.vocab = nn.Embedding(vocab_size, d_model)
		transformer_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=2048,
			dropout=0.1,
			activation="relu",
			layer_norm_eps=0.00001,
			batch_first=True,
			norm_first=True,
			bias=True,
		)
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
	grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

	optimizer.step()

    # 参数变化证据
	weight_delta = (model.lm_head.weight.detach() - old_weight).abs().mean()
	metrics = {"loss":loss.item(), "grad_norm":grad_norm, "lm_head_delta":weight_delta.item()}
	return metrics

def evaluate_single_batch(
		model,
		input_ids,
		labels,
		attention_mask
):
	was_training = model.training
	model.eval()
	with torch.no_grad():
		logits = model(input_ids, attention_mask)
		shift_logits = logits[:, :-1, :]
		shift_labels = labels[:, 1:]
		mask = shift_labels != -100
		valid_tokens = mask.sum()
		if valid_tokens == 0:
			raise ValueError("No valid target tokens found: all labels are ignored by ignore_index=-100")
		loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100, reduction="mean")
		cls = torch.argmax(shift_logits, dim=2)
		correct_tokens = torch.sum(cls[mask] == shift_labels[mask])
		token_accuracy = correct_tokens / valid_tokens if valid_tokens != 0  else 0

	if was_training:
		model.train()
	return loss, correct_tokens, valid_tokens, token_accuracy

def overfit():
	model = TinyCausalLM()
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
	
	locations = ["BOS", "SYS", "USER", "ASSIST", "A", "B", "EOS", "PAD"]
	input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]]) # [1, 8]
	attention = torch.tensor([[1, 1, 1, 1, 1, 1 ,1, 0]]) # [1, 8]
	labels = torch.tensor([[-100, -100, -100, -100, 5, 6, 7, -100]])

	max_epoch = 1000
	loss_thresh = 0.1

	for i in range(max_epoch):
		metrics = train_step(model, optimizer=optimizer, input_ids=input_ids, labels=labels, attention_mask=attention)
		train_loss = metrics["loss"]
		if train_loss < loss_thresh:
			eval_loss, _, _, accuracy = evaluate_single_batch(model, input_ids=input_ids, labels=labels, attention_mask=attention)
			if eval_loss < loss_thresh and accuracy >= 0.99:
				print(f"overfit:(loss:{eval_loss.item()}),(accuray:{accuracy.item()})")
				return True

	print("max epoch arrived")


'''
 K = 20，M = 5
  1. 训练 K 步并保存 checkpoint
  2. 原分支继续 M 步，保存每步 metrics
  3. 克隆原分支最终参数
  4. 新建并恢复 model/optimizer/RNG
  5. 恢复分支训练 M 步
  6. 输出：
     - max_loss_diff
     - max_grad_norm_diff
     - max_lm_head_delta_diff
     - max_param_abs_diff

'''
def recover_trajectory():
	model = TinyCausalLM()
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

	locations = ["BOS", "SYS", "USER", "ASSIST", "A", "B", "EOS", "PAD"]
	input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]]) # [1, 8]
	attention = torch.tensor([[1, 1, 1, 1, 1, 1 ,1, 0]]) # [1, 8]
	labels = torch.tensor([[-100, -100, -100, -100, 5, 6, 7, -100]])

	K = 20
	M = 5

	for _ in range(K):
		train_step(model, optimizer=optimizer, input_ids=input_ids, labels=labels,  attention_mask=attention)
	checkpoint = {
		"model": copy.deepcopy(model.state_dict()),
		"optimizer": copy.deepcopy(optimizer.state_dict()),
		"RNG": copy.deepcopy(torch.get_rng_state())
	}

	reference_metrics = []

	for _ in range(M):
		metrics = train_step(model, optimizer=optimizer, input_ids=input_ids, labels=labels,  attention_mask=attention)
		reference_metrics.append(metrics)

	reference_params = {name: param.detach().clone() for name, param in model.named_parameters()}

	# resume training

	model = TinyCausalLM()
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
	model.load_state_dict(checkpoint["model"])
	optimizer.load_state_dict(checkpoint["optimizer"])
	torch.set_rng_state(checkpoint["RNG"])

	max_loss_diff = 0
	max_grad_norm_diff = 0
	max_lm_head_delta_diff = 0
	for i in range(M):
		metrics = train_step(model, optimizer=optimizer, input_ids=input_ids, labels=labels,  attention_mask=attention)
		max_loss_diff = max(max_loss_diff, abs(metrics["loss"] - reference_metrics[i]["loss"]))
		max_grad_norm_diff = max(max_grad_norm_diff, abs(metrics["grad_norm"] - reference_metrics[i]["grad_norm"]))
		max_lm_head_delta_diff = max(max_lm_head_delta_diff, abs(metrics["lm_head_delta"] - reference_metrics[i]["lm_head_delta"]))
	max_param_abs_diff = 0
	for name, param in model.named_parameters():
		diff = (param.detach() - reference_params[name]).abs().max()
		max_param_abs_diff = max(max_param_abs_diff, diff)
	print(max_loss_diff, max_grad_norm_diff, max_lm_head_delta_diff, max_param_abs_diff)
	

if __name__ == "__main__":
	recover_trajectory()



