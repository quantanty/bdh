# Copyright Pathway Technology, Inc.

import os
from argparse import ArgumentParser
from contextlib import nullcontext

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

import bdh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


# Configuration
BDH_CONFIG = bdh.BDHConfig()
BLOCK_SIZE = 512
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
MAX_ITERS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 10
CHECKPOINT_FREQ = 100

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
output_dir = "v0"


# Fetch the tiny Shakespeare dataset
def fetch_data():
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)


def get_batch(split, block_size, batch_size):
    # treat the file as bytes
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)) :]
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if torch.cuda.is_available():
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def eval(model):
    model.eval()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE, required=False)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, required=False)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=GRADIENT_ACCUMULATION_STEPS, required=False)
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS, required=False)
    parser.add_argument("--log-freq", type=int, default=LOG_FREQ, required=False)
    parser.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQ, required=False)
    parser.add_argument("--output-dir", type=str, default=output_dir, required=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    block_size = args.block_size
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    max_iters = args.max_iters
    log_freq = args.log_freq
    checkpoint_freq = args.checkpoint_freq
    output_dir = args.output_dir

    fetch_data()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model = bdh.BDH(BDH_CONFIG).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    x, y = get_batch("train", block_size, batch_size)

    for step in range(max_iters):
        loss_acc = 0
        for micro_step in range(gradient_accumulation_steps):
            x, y = get_batch("train", block_size, batch_size)
            with ctx:
                logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
            loss_acc += loss
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if step % log_freq == 0:
            print(f"Step: {step}/{max_iters} loss {loss_acc:.3}")
        if output_dir and step % checkpoint_freq == 0 and step > 0:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step}, f'{output_dir}/checkpoint_{step}.pt')
    if output_dir:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': max_iters}, f'{output_dir}/final_checkpoint.pt')
    print("Training done, now generating a sample ")
    model.eval()
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)
