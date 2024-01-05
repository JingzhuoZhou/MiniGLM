import os
import torch
import numpy as np

train_data = None
val_data = None
train_loss_mask=None
val_loss_mask=None

def init_data_pretrain(dataset):
    global train_data, val_data
    
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def init_data_sft(dataset):
    global train_data, val_data, train_loss_mask,val_loss_mask
    
    ### 读取+初始化sft数据
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    train_loss_mask = np.memmap(os.path.join(data_dir, 'train_loss.bin'), dtype=np.uint16, mode='r')
    val_loss_mask = np.memmap(os.path.join(data_dir, 'val_loss.bin'), dtype=np.uint16, mode='r')
    ###

def get_batch_pretrain(split, batch_size, block_size, device):
    global train_data, val_data
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    loss_mask = torch.ones_like(x, dtype=torch.float64)
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    return x, y, loss_mask
    
def get_batch_sft(split, batch_size, block_size, device): 
    ### 获取sft数据的批次（batch）+ 构建损失函数掩码（loss_mask）
    global train_data, val_data,train_loss_mask,val_loss_mask
    data = train_data if split == 'train' else val_data
    mask=train_loss_mask if split=='train' else val_loss_mask
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    loss_mask = torch.stack([torch.from_numpy((mask[i:i+block_size]).astype(np.int64)) for i in ix])
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    
    ###
    
    return x, y, loss_mask