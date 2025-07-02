import os
import json
import math
import argparse
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import load_dataset, load_from_disk, concatenate_datasets
from bitnet import replace_linears_in_hf as apply_bitnet
from layers.compbitlinear import replace_linears_with_complexbitnet

import torch
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM

class CompGPTNeoForCausalLM(GPTNeoForCausalLM):
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Embedding留??ㅼ닔濡??좎??섍퀬 蹂듭냼?섎줈 罹먯뒪??        if input_ids is not None and kwargs.get("inputs_embeds", None) is None:
            inputs_embeds = self.transformer.wte(input_ids)
            inputs_embeds = inputs_embeds.to(torch.complex64)
            kwargs["inputs_embeds"] = inputs_embeds

        return super().forward(input_ids=None, attention_mask=attention_mask, **kwargs)

def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def get_model_and_tokenizer(model_name, model_type):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config,torch_dtype=torch.bfloat16)#, attn_implementation="flash_attention_2")
    
    if model_type == "bitnet":
        apply_bitnet(model)
    elif model_type == "compbitnet":
        model = CompGPTNeoForCausalLM.from_config(config,torch_dtype=torch.bfloat16)
        replace_linears_with_complexbitnet(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def tokenize_dataset(tokenizer, tokenized_path, max_length):
    
    if not os.path.exists(tokenized_path):
        raw_dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")

        def tokenize_fn(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

        tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, num_proc=64, remove_columns=["text"])
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        tokenized_dataset.save_to_disk(tokenized_path)
    
    dataset = load_from_disk(tokenized_path)
    return dataset

def load_datasets():
    configs = ['arxiv', 'c4', 'common_crawl', 'github', 'stackexchange', 'wikipedia']
    # configs = ['arxiv']
    datasets = [load_from_disk(f"/nvme_pool/redpajama-data-v1/gpt-neo-125M-tokenized/{cfg}")['train'] for cfg in configs]
    dataset = concatenate_datasets(datasets)
    return dataset

def prepare_datasets(dataset, validation_split):
    split_idx = int(len(dataset) * (1 - validation_split))
    train_dataset = dataset
    eval_dataset = dataset.select(range(split_idx, len(dataset)))

    # def add_labels(example):
    #     example["labels"] = example["input_ids"]
    #     return example

    # train_dataset = train_dataset.map(add_labels, num_proc=16)
    # eval_dataset = eval_dataset.map(add_labels, num_proc=16)
    return train_dataset, eval_dataset


def main(config_path):
    config = load_config(config_path)
    os.makedirs(config["output_dir"], exist_ok=True)

    model, tokenizer = get_model_and_tokenizer(config["model_name"], config["model_type"])
    # dataset = tokenize_dataset(tokenizer, config["tokenized_dataset_path"], config["max_length"])
    dataset = load_datasets()
    train_dataset, eval_dataset = prepare_datasets(dataset, config["validation_split"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum_steps"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        weight_decay=config["weight_decay"],
        bf16=config["bf16"],
        logging_steps=10,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=2,
        eval_strategy="steps",
        logging_dir=os.path.join(config["output_dir"], "logs"),
        dataloader_num_workers=0,
        report_to="none",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    final_metrics = {
        "loss": eval_results["eval_loss"],
        "perplexity": math.exp(eval_results["eval_loss"]),
        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
        "epoch": training_args.num_train_epochs,
        "grad_norm": trainer.optimizer.param_groups[0].get("grad_norm", None),
    }

    with open(os.path.join(config["output_dir"], "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    print("??Final Metrics:", final_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    args = parser.parse_args()
    main(args.config)
