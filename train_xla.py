# train_gpt_neox_ddp.py
import os
import json
import argparse
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import numpy as np

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.distributed.xla_backend
import time
from datetime import datetime, timedelta

import torch_xla.experimental.pjrt_backend  # register PJRT backend
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def _mp_fn(index,args):

    # Initialize XLA process group
    dist.init_process_group(backend='xla', init_method='xla://')
    device = 'xla'
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ('data', 'model'))


    # Load config
    cfg = json.load(open(args.config))
    model_name   = cfg['model_name']
    batch_size   = cfg.get('batch_size', 8)
    max_length   = cfg.get('max_length', 512)
    epochs       = cfg.get('epochs', 1)
    lr           = cfg.get('learning_rate', 5e-5)

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    ddp_model = DDP(model, gradient_as_bucket_view=True)
    # Prepare RedPajama dataset
    ds = load_dataset('togethercomputer/RedPajama-Data-1T-Sample', split='train')
    def tokenize_fn(ex):
        return tokenizer(ex['text'], truncation=True, max_length=max_length, padding='max_length')
    ds = ds.map(tokenize_fn, batched=True, remove_columns=['text'], num_proc=12)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Shard dataset per process
    # world_size = xm.xrt_world_size()
    # ds = ds.shard(num_shards=world_size, index=rank)
    # print(mesh.shape())
    # sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=True,
        num_workers=8,
        pin_memory=False
    )
    total_steps = len(loader)
    mp_device_loader = pl.MpDeviceLoader(loader, device, input_sharding={
          'input_ids': xs.ShardingSpec(mesh, ('data', None)),
          'attention_mask': xs.ShardingSpec(mesh, ('data', None))
        })

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()
        step = 0
        model.train()
        total_loss = 0.0
        for batch in iter(mp_device_loader):
            step = step + 1
            step_loss = 0.0
            with torch_xla.step():
                inputs = batch['input_ids'].to(device)
                masks  = batch['attention_mask'].to(device)
                optimizer.zero_grad()
                outputs = model(inputs, attention_mask=masks, labels=inputs)
                loss = outputs.loss
                loss.backward()
                xm.optimizer_step(optimizer)
                total_loss += loss.item()
                if step % 10 == 0:
                    elapsed = time.time() - epoch_start
                    avg_per_step = elapsed / (step + 1)
                    remaining = total_steps - (step + 1)
                    eta_sec = avg_per_step * remaining
                    eta_str = str(timedelta(seconds=int(eta_sec)))
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"Epoch {epoch+1}/{epochs}, Step {step}/{total_steps}, "
                        f"Loss={step_loss/10:.4f}, Elapsed={elapsed:.1f}s, ETA={eta_str}")
                    step_loss = 0.0
        if xm.is_master_ordinal():
            print(f"Epoch {epoch} avg loss: {total_loss/len(loader):.4f}")

    if xm.is_master_ordinal():
        print("✅ Training complete")


def main():
    # xr.use_spmd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON config with model_name, batch_size, max_length, epochs, learning_rate')
    args = parser.parse_args()
    torch_xla.launch(_mp_fn, args=(args,))

if __name__ == '__main__':
    main()
