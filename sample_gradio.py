import random
import gradio as gr
import os
from contextlib import nullcontext
import torch
import tiktoken
from model import GLMConfig, MiniGLM
import re

# -----------------------------------------------------------------------------

out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 512 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1234
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file

# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
config = GLMConfig(**checkpoint['model_args'])
model = MiniGLM(config)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

gpt2 = tiktoken.get_encoding("gpt2")
enc=tiktoken.Encoding(
    name='gpt2',
    pat_str=gpt2._pat_str,
    mergeable_ranks=gpt2._mergeable_ranks,
    special_tokens={
        '<|endoftext|>':50256,
        '<|delimiter|>':50527
    }
)
encode = lambda s: enc.encode(s, allowed_special='all')
decode = lambda l: enc.decode(l)


def respond(message, history):
    message=message+'\n'# 识别问题

    # 统一乔峰萧峰
    flag=False
    if(message.find('乔峰')!=-1):
        message=message.replace('乔峰','萧峰')
        flag=True
    
    start_ids = encode(message)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    output=''
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                output_tokens = y[0].tolist()
                try:
                    end_idx = output_tokens.index(50256)
                    output_tokens = output_tokens[:end_idx]
                except:
                    pass

                output = output+decode(output_tokens)

    # 提取答案
    if(output.find('答：')!=-1):
        output=output[output.find('答：')+2:]

    # 删除乱码
    output=output.replace('\ufffd','')

    # 还原乔峰萧峰为提问的形式
    if(flag):
        output=output.replace('萧峰','乔峰')
    return output

gr.ChatInterface(respond).launch()