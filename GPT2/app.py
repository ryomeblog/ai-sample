import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# トークナイザのロード
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

# モデルの設定をロード
config = GPT2Config()

# モデルのインスタンスを作成
model = GPT2LMHeadModel(config)

# 重みをロード
model_weights = torch.load("/usr/src/data/ggml-model.bin", map_location="cpu")
model.load_state_dict(model_weights)
model.eval()

# 生成したい文章のプロンプト
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# モデルを使って文章を生成
output = model.generate(input_ids, max_length=100)

# 生成された文章を表示
decoded_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_sequence)
