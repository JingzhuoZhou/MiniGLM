import os
import sys
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

names = sys.argv[1:]

all_texts = []

# Read data from ([name]/input.txt for name in names)
for name in names:
    file_path = os.path.join(name, "input.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        all_texts.append(text)

# Combine multiple books into one single data file
combined_text = "".join(all_texts)

# Split data for train (90%) and valid (10%)
total_len = len(combined_text)
train_len = int(0.9 * total_len)
train_data = combined_text[:train_len]
val_data = combined_text[train_len:]

# Tokenize raw data with tiktoken encoder
train_tokens = enc.encode_ordinary(train_data)
val_tokens = enc.encode_ordinary(val_data)

# Transform tokenized data to numpy array
train_ids = np.array(train_tokens,dtype=np.uint16)
val_ids = np.array(val_tokens,dtype=np.uint16)

# Save numpy array to files out_dir/train.bin and out_dir/val.bin
output_dir=''
for name in names:
    output_dir = output_dir+name[0]
os.makedirs(output_dir, exist_ok=True)
train_ids.tofile(os.path.join(output_dir, "train.bin"))
val_ids.tofile(os.path.join(output_dir, "val.bin"))
