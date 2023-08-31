import torch, torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

torch.npu.set_compile_mode(jit_compile=False)

model_path = "/home/username/EXPORT_MODEL/qwen-7b-chat/pt/"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="npu:0", trust_remote_code=True,fp16=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

history = None
while True:
    quary = input("input:")
    if quary == "exit":
        break
    response, history = model.chat(tokenizer, quary, history=None)
    print(f"response:{response}")
