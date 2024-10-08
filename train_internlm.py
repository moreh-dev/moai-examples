from argparse import ArgumentParser
import copy
import os
import sys
import time

from datasets import load_dataset
from loguru import logger
import torch
from transformers import AdamW
from transformers import AutoConfig
from transformers import AutoTokenizer

from model.internlm.modeling_internlm2 import InternLM2ForCausalLM
from utils import *


# Arguments
def parse_args():
    parser = ArgumentParser(description="InternLM FineTuning")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="internlm/internlm2_5-20b-chat",
        help="model name or path",
    )
    parser.add_argument("--epochs",
                        type=int,
                        default=1,
                        help="num training epochs")
    parser.add_argument("--train-batch-size",
                        type=int,
                        default=16,
                        help="train bacth size")
    parser.add_argument("--eval-batch-size",
                        type=int,
                        default=1,
                        help="evaluation bacth size")
    parser.add_argument("--block-size",
                        type=int,
                        default=32768,
                        help="max input token length")
    parser.add_argument(
        "--dataset-name-or-path",
        type=str,
        default=
        "agileloop/izaz-sequence-of-actions-prediction-dataset-llama2-7b-32k",
        help="dataset name or path",
    )
    parser.add_argument("--lr",
                        type=float,
                        default=0.00001,
                        help="learning rate")
    parser.add_argument("--log-interval",
                        type=int,
                        default=20,
                        help="log interval")
    parser.add_argument(
        "--eval-step",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./internlm2_5-20b-finetuned",
        help="model save path",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
    )
    args = parser.parse_args()
    return args


def eval(model, eval_dataloader, tokenizer):
    with torch.no_grad():
        logger.info("[START EPOCH EVAL]")
        model.eval()
        ev_st = time.time()
        eval_loss = torch.tensor([0], device="cuda")
        total_correct = torch.tensor([0], device="cuda")
        for e_step, e_batch in enumerate(eval_dataloader, start=1):
            e_input_ids = e_batch['input_ids']
            e_attn_mask = e_batch['attenion_mask']
            e_inputs, e_labels = e_input_ids, mask_pads(e_input_ids,
                                                        e_attn_mask)
            e_position_ids = e_attn_mask.long().cumsum(-1) - 1
            e_position_ids.masked_fill_(e_attn_mask == 0, 1)
            if e_step % 10 == 0:
                logger.info(f"EVAL STEP: {e_step} / {len(eval_dataloader)}")
            e_outputs = model(
                e_inputs.cuda(),
                attention_mask=e_attn_mask,
                position_ids=e_position_ids,
                labels=e_labels.cuda(),
                use_cache=False,
            )
            eval_loss += e_outputs[0]
        logger.info(f"EVAL STEP: {e_step} / {len(eval_dataloader)}")
        logger.info(
            f"Eval Loss: {eval_loss.item()/len(eval_dataloader)} | ELAPSED EVAL TIME: {(time.time() - ev_st)} sec"
        )


def main(args):
    torch.moreh.option.enable_advanced_parallelization()
    # Load base model and tokenizer
    print(f"Load {args.model_name_or_path} model checkpoint and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path,
                                        trust_remote_code=True)
    model = InternLM2ForCausalLM.from_pretrained(args.model_name_or_path,
                                                 trust_remote_code=True)
    if args.use_lora:
        from peft import get_peft_model
        from peft import LoraConfig
        config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    print_trainable_parameters(model)
    print(f"Downloading {args.dataset_name_or_path} dataset...")
    dataset = load_dataset(args.dataset_name_or_path).with_format("torch")
    if "validation" not in dataset:
        dataset["train"] = load_dataset(
            args.dataset_name_or_path, split="train[:90%]").with_format("torch")
        dataset["validation"] = load_dataset(
            args.dataset_name_or_path,
            split="train[90%:95%]").with_format("torch")

    # Construct a formatted prompt
    def preprocess(prompt):
        chat = [{
            "role": "user",
            "content": f"{prompt['Instruction']}"
        }, {
            "role": "assistant",
            "content": f"{prompt['Response']}"
        }]

        chat = tokenizer.apply_chat_template(chat, tokenize=False)
        result = tokenizer(chat,
                           truncation=True,
                           max_length=args.block_size,
                           padding="max_length")
        ret = {}
        ret['input_ids'] = result['input_ids']
        ret['attention_mask'] = result['attention_mask']
        return ret

    def collator(data):
        return {
            'input_ids': torch.stack([x['input_ids'] for x in data]),
            'attention_mask': torch.stack([x['attention_mask'] for x in data])
        }

    dataset = dataset.map(preprocess, num_proc=8)

    # Create a DataLoader for the training set
    # Use collate_fn to ensure that all data samples are of the sample length
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
    )

    # Use collate_fn to ensure that all data samples are of the sample length
    # Create a DataLoader for the validation set
    eval_dataloader = torch.utils.data.DataLoader(
        dataset["validation"],
        batch_size=args.eval_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator)

    # Prepare the model for training on Accelerator
    model.cuda()
    model.train()
    # Define AdamW optimizer
    optim = AdamW(model.parameters(), lr=args.lr)
    # Calculate total training steps
    total_step = len(train_dataloader) * args.epochs
    # Start training
    for epoch in range(args.epochs):
        st = time.time()
        for step, batch in enumerate(train_dataloader, start=1):
            start_time = time.perf_counter()
            input_ids = batch['input_ids']
            attn_mask = batch['attention_mask']
            inputs, labels = input_ids, mask_pads(input_ids, attn_mask)
            position_ids = attn_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attn_mask == 0, 1)
            outputs = model(
                input_ids.cuda(),
                attention_mask=attn_mask,
                labels=labels.cuda(),
                position_ids=position_ids,
                use_cache=False,
            )
            loss = outputs[0]
            loss.backward()
            optim.step()
            model.zero_grad(set_to_none=True)
            # Logging
            if step == 1:
                loss.item()
                logger.info(
                    f"Model load and warmup done. Duration: {(time.time() - st):.2f}"
                )
                st = time.time()
                continue
            if step % args.log_interval == 0:
                if step == args.log_interval:
                    step_interval = args.log_interval - 1
                else:
                    step_interval = args.log_interval
                logger.info(
                    f"[Step {step+(epoch*len(train_dataloader))}/{total_step}] | Loss: {loss.item()} | Duration: {(time.time() - st):.2f} | {((step_interval * args.batch_size)/(time.time() - st)):.2f} | Throughput: {((step_interval * args.batch_size * args.block_size)/(time.time() - st)):.2f} tokens/sec"
                )
                st = time.time()
            if step % args.eval_step == 0:
                # Evaluation
                eval(model, eval_dataloader, tokenizer)
                model.train()
                st = time.time()
        # Evaluation
        eval(model, eval_dataloader, tokenizer)
        model.train()
        st = time.time()
    save_model(args, model, tokenizer)


if __name__ == "__main__":
    args = parse_args()
    main(args)
