import os
import tiktoken
import numpy as np
import json

# Get encoder
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

dic=[]# sft data
f=open('sft_data.jsonl','r',encoding='utf-8')
for l in f:
    dic.append(json.loads(l))
f.close()

# train data 90% & val data 10%
n=len(dic)
train_num=int(0.9*n)

train_tokens=[]
train_loss=[]# loss mask of train data

for i in range(train_num):
    data=dic[i]
    text='问：'+data['Question'] +'\n'
    tokens = enc.encode(text,allowed_special={"<|endoftext|>"})
    loss=[0 for x in range(len(tokens)-1)]+[1]# Set loss mask = 0 for questions & 1 for delimeter('\n')
    
    text='答：'+data['Answer']
    tks=enc.encode(text,allowed_special={"<|endoftext|>"})
    tokens = tokens+tks
    loss=loss+[1 for x in range(len(tks))]# count loss for answers
    
    if(len(loss)>511):# Truncate the answer
        tokens=tokens[:511]
        loss=loss[:511]

    tokens.append(50256)# Padding (<|endoftext|>)
    loss.append(1)# Count the loss of the first padding
    for j in range(len(tokens),512):# Padding to block size
        tokens.append(50256)
        loss.append(0)
    
    train_tokens=train_tokens+tokens
    train_loss=train_loss+loss
    

val_tokens=[]
val_loss=[]
for i in range(train_num,n):
    data=dic[i]
    text='问：'+data['Question'] +'\n'
    tokens = enc.encode(text,allowed_special={"<|endoftext|>"})
    loss=[0 for x in range(len(tokens)-1)]+[1]

    text='答：'+data['Answer']
    tks=enc.encode(text,allowed_special={"<|endoftext|>"})
    tokens = tokens+tks
    loss=loss+[1 for x in range(len(tks))]
    
    if(len(loss)>511):
        tokens=tokens[:511]
        loss=loss[:511]
    
    tokens.append(50256)
    loss.append(1)
    for j in range(len(tokens),512):
        tokens.append(50256)
        loss.append(0)
    
    val_tokens=val_tokens+tokens
    val_loss=val_loss+loss

# Transform tokenized data to numpy array
train_ids = np.array(train_tokens,dtype=np.uint16)
val_ids = np.array(val_tokens,dtype=np.uint16)
trainloss_ids = np.array(train_loss,dtype=np.uint16)
valloss_ids = np.array(val_loss,dtype=np.uint16)

# Save numpy array to binary files
output_dir='sft'
os.makedirs(output_dir, exist_ok=True)
train_ids.tofile(os.path.join(output_dir, "train.bin"))
val_ids.tofile(os.path.join(output_dir, "val.bin"))
trainloss_ids.tofile(os.path.join(output_dir, "train_loss.bin"))
valloss_ids.tofile(os.path.join(output_dir, "val_loss.bin"))