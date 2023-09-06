# LLM Efficient Tuning on NPU  
此文档演示了如何在NPU上进行大模型的训练。  
 
## Prepare the Model  
[Qwen-7b-chat](https://huggingface.co/Qwen/Qwen-7B-Chat)  
下载模型pytorch权重，保存到路径`model_path_and_name = /home/username/MODEL`  

## Prepare the environment  
```shell
cd /home/username
create -n qwen python=3.8
conda activate qwen
pip install torch
pip install torch_npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh

git clone -b npu https://github.com/duweidongzju/LLaMA-Efficient-Tuning.git
cd ./LLaMA-Efficient-Tuning
pip install -r requirements.txt
pip install google protobuf decorator absl-py cloudpickle synr==0.5.0 tornado einops transformers_stream_generator
```

## Train

训练前，需要在 `src/train_bash.py` 文件中添加代码  
```python
import torch
import torch_npu
```

使用以下脚本进行训练  
```bash
#!/bin/bash
model_name=qwen-7b-chat
path_to_llama_model=/home/username/MODEL/$model_name
path_to_pt_checkpoint=/home/username/OUTPUT/$model_name/pt

mkdir -p $path_to_pt_checkpoint

INF_NAN_MODE_ENABLE=1 python src/train_bash.py \
    --stage pt \
    --model_name_or_path $path_to_llama_model \
    --do_train \
    --dataset wiki_demo \
    --template chatml \
    --finetuning_type lora \
    --lora_target c_attn \
    --output_dir $path_to_pt_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --fp16 \
    --overwrite_output_dir
```

<details>
<summary> NPU 多卡训练流程 </summary>

设置 accelerate config
```shell
>>> accelerate config
-------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
-> This machine                                                                                                                              
-------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?                                                                                                                                       
-> multi-NPU                                                                                                                                                
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1                                                                                              
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: no                                                                
Do you wish to optimize your script with torch dynamo?[yes/NO]:no                                                                         
Do you want to use FullyShardedDataParallel? [yes/NO]: no                                                                                                                  
How many NPU(s) should be used for distributed training? [1]:8                                                                                                                   
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:all
--------------------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?
-> fp16
```

使用下面脚本进行多卡训练,脚本路径`./scripts/run_pretraining_multi_NPUs.sh`  
如果需要训练其它模型。请修改`--model_name_or_path`,`--lora_target`,`--template`  
```bash
#!/bin/bash

model_name=llama-2-7b-hf
path_to_llama_model=/home/username/MODEL/$model_name
path_to_pt_checkpoint=/home/username/CHECKPOINT/$model_name/pt
mkdir -p path_to_pt_checkpoint

INF_NAN_MODE_ENABLE=1 accelerate launch scr/train_bash.py
    --stage pt \
    --model_name_or_path $path_to_llama_model \
    --do_train \
    --dataset wiki_domo \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir $path_to_pt_checkpoint \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 10.0 \
    --fp16
```

</details>

## Export trained Model  
导出模型前，需要在 `src/export_model.py` 文件中添加代码  
```python
import torch
import torch_npu
```

运行下列脚本导出模型  
```bash
#!/bin/bash
  
model_name=qwen-7b-chat
path_to_llama_model=/home/username/MODEL/$model_name
path_to_checkpoint=/home/username/OUTPUT/$model_name/pt
path_to_export=/home/username/EXPORT_MODEL/$model_name/pt

python src/export_model.py \
    --model_name_or_path $path_to_llama_model \
    --template chatml \
    --finetuning_type lora \
    --checkpoint_dir $path_to_checkpoint \
    --output_dir $path_to_export
```

## Inference
运行下列python代码进行在线对话
```python
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
```

some chat examples:  
```shell
input:你好
response:你好！很高兴为你提供帮助。

input:web应用防火墙(waf)可以防火吗？
response:是的，Web应用防火墙（WAF）是一种网络安全技术，可以防止Web应用程序受到攻击和滥用。它可以检测和阻止针对Web应用程序的攻击，例如SQL注入、跨站脚本攻击（XSS）和跨站请求伪造（CSRF）等。WAF通常使用规则引擎和机器学习技术来识别攻击，并采取适当的措施来阻止它们。

input:请介绍迷网和蜜罐的区别
response:迷网和蜜罐都是网络安全领域中的概念，它们的主要区别在于它们的目的和作用。
迷网（也称为欺骗网络）是一种攻击技术，其目的是通过制造虚假的网络环境来欺骗攻击者或恶意软件。这种技术通常使用欺骗性的IP地址、虚假的网络服务、伪造的认证信息等手段来制造虚假的网络环境。攻击者在进入迷网后，会误认为自己处于一个安全的环境中，从而放松警惕，进而被攻击者利用。
蜜罐（也称为诱饵系统）是一种网络安全技术，其目的是通过故意暴露一些虚假的信息或资源来吸引攻击者。这种技术通常使用一些看起来像是真实目标的网络服务或资源，如Web服务器、电子邮件服务器、数据库等，来吸引攻击者。攻击者在访问这些蜜罐时，会向其暴露一些有价值的信息或资源，从而被攻击者捕获和分析。
总之，迷网和蜜罐都是网络安全领域中的技术，它们的目的和作用不同。迷网旨在欺骗攻击者，而蜜罐旨在吸引攻击者并捕获其信息。

input:exit
```

# Support Models

## model
| 模型名                                                   | 模型大小                    | lora_target       | Template |
| -------------------------------------------------------- | --------------------------- | ----------------- |----------|
| [LLaMA](https://github.com/facebookresearch/llama)       | 7B/13B/33B/65B              | q_proj,v_proj     | -        |
| [LLaMA-2](https://huggingface.co/meta-llama)             | 7B/13B/70B                  | q_proj,v_proj     | llama2   |
| [BLOOM](https://huggingface.co/bigscience/bloom)         | 560M/1.1B/1.7B/3B/7.1B/176B | query_key_value   | -        |
| [BLOOMZ](https://huggingface.co/bigscience/bloomz)       | 560M/1.1B/1.7B/3B/7.1B/176B | query_key_value   | -        |
| [Falcon](https://huggingface.co/tiiuae/falcon-7b)        | 7B/40B                      | query_key_value   | -        |
| [Baichuan](https://github.com/baichuan-inc/baichuan-13B) | 7B/13B                      | W_pack            | baichuan |
| [InternLM](https://github.com/InternLM/InternLM)         | 7B                          | q_proj,v_proj     | intern   |
| [Qwen](https://github.com/QwenLM/Qwen-7B)                | 7B                          | c_attn            | chatml   |
| [XVERSE](https://github.com/xverse-ai/XVERSE-13B)        | 13B                         | q_proj,v_proj     | xverse   |

- **默认模块**是 `--lora_target` 参数的部分可选项。请使用 `python src/train_bash.py -h` 查看全部可选项。
- 对于所有“基座”（Base）模型，`--template` 参数可以是 `default`, `alpaca`, `vicuna` 等任意值。但“对话”（Chat）模型请务必使用对应的模板。

