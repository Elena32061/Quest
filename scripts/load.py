from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型名称
model_name = "lmsys/longchat-7b-v1.5-32k"

# 下载并加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 保存模型到指定路径
save_directory = "/home/jiangnanyu/questreproduce/Quest/dataset_model/longchat-7b-v1.5-32k"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
