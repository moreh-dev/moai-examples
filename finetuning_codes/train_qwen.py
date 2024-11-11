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
from utils import *


# Arguments
def parse_args():
    parser = ArgumentParser(description="Qwen FineTuning")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Qwen/Qwen-14B",
        help="model name or path",
    )
    parser.add_argument("--epochs",
                        type=int,
                        default=1,
                        help="num training epochs")
    parser.add_argument("--train-batch-size",
                        type=int,
                        default=16,
                        help="train batch size")
    parser.add_argument("--eval-batch-size",
                        type=int,
                        default=1,
                        help="evaluation batch size")
    parser.add_argument("--block-size",
                        type=int,
                        default=1024,
                        help="max input token length")
    parser.add_argument(
        "--dataset-name-or-path",
        type=str,
        default=
        "MBZUAI/LaMini-instruction",
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
        "--max-step",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./qwen-finetuned",
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


def eval(args, model, eval_dataloader, tokenizer):
    with torch.no_grad():
        logger.info("[START EPOCH EVAL]")
        model.eval()
        ev_st = time.time()
        eval_loss = torch.tensor([0], device="cuda")
        total_correct = torch.tensor([0], device="cuda")
        for e_step, e_batch in enumerate(eval_dataloader, start=1):
            e_input_ids = e_batch['input_ids']
            e_attn_mask = e_batch['attention_mask']
            e_inputs, e_labels = e_input_ids, mask_pads(e_input_ids,
                                                        e_attn_mask)
            e_position_ids = e_attn_mask.long().cumsum(-1) - 1
            e_position_ids.masked_fill_(e_attn_mask == 0, 1)
            if e_step % 10 == 0:
                logger.info(f"EVAL STEP: {e_step} / {len(eval_dataloader)}")
            e_outputs = model(
                e_inputs.cuda(),
                attention_mask=e_attn_mask.cuda(),
                position_ids=e_position_ids.cuda(),
                labels=e_labels.cuda(),
                use_cache=False,
            )
            eval_loss += e_outputs[0]
            if args.max_step == e_step:
                break
        logger.info(f"EVAL STEP: {e_step} / {len(eval_dataloader)}")
        logger.info(
            f"Eval Loss: {eval_loss.item()/len(eval_dataloader)} | ELAPSED EVAL TIME: {(time.time() - ev_st)} sec"
        )


def main(args):
    torch.moreh.option.enable_advanced_parallelization()
    # Load base model and tokenizer
    model, tokenizer = load_model(args)

    print(f"Downloading {args.dataset_name_or_path} dataset...")
    dataset = load_custom_dataset(args)
    dataset = preprocess_dataset(args, dataset, tokenizer)

    def collator(data):
        return {
            'input_ids': torch.stack([x['input_ids'] for x in data]),
            'attention_mask': torch.stack([x['attention_mask'] for x in data])
        }

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
        dataset["test"],
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
    current_step = 0
    # Start training
    st = time.time()
    for epoch in range(args.epochs):
        for _, batch in enumerate(train_dataloader, start=1):
            current_step += 1
            input_ids = batch['input_ids']
            attn_mask = batch['attention_mask']
            labels = mask_pads(input_ids, attn_mask)
            position_ids = attn_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attn_mask == 0, 1)
            outputs = model(
                input_ids.cuda(),
                attention_mask=attn_mask.cuda(),
                labels=labels.cuda(),
                use_cache=False,
            )
            loss = outputs[0]
            loss.backward()
            optim.step()
            model.zero_grad(set_to_none=True)
            # Logging
            if args.max_step != -1 and current_step == args.max_step:
                if current_step % args.log_interval == 0:
                    if current_step == args.log_interval:
                        step_interval = args.log_interval - 1
                    else:
                        step_interval = args.log_interval
                    logger.info(
                        f"[Step {current_step}/{total_step}] | Loss: {loss.item()} | Duration: {(time.time() - st):.2f} | Throughput: {((step_interval * args.train_batch_size * args.block_size)/(time.time() - st)):.2f} tokens/sec"
                    )
                else:
                    step_interval = current_step % args.log_interval
                    logger.info(
                        f"[Step {current_step}/{total_step}] | Loss: {loss.item()} | Duration: {(time.time() - st):.2f} | Throughput: {((step_interval * args.train_batch_size * args.block_size)/(time.time() - st)):.2f} tokens/sec"
                    )
                eval(args, model, eval_dataloader, tokenizer)
                break
            if current_step == 1:
                loss_item = loss.item()
                logger.info("Model load and 1 step done.")
                logger.info(
                    f"[Step {current_step}/{total_step}] | Loss: {loss_item} | Duration: {(time.time() - st):.2f}"
                )
                st = time.time()
                continue
            if current_step % args.log_interval == 0:
                if current_step == args.log_interval:
                    step_interval = args.log_interval - 1
                else:
                    step_interval = args.log_interval
                logger.info(
                    f"[Step {current_step}/{total_step}] | Loss: {loss.item()} | Duration: {(time.time() - st):.2f} | Throughput: {((step_interval * args.train_batch_size * args.block_size)/(time.time() - st)):.2f} tokens/sec"
                )
                st = time.time()
            if current_step % args.eval_step == 0:
                # Evaluation
                eval(args, model, eval_dataloader, tokenizer)
                model.train()
                st = time.time()

        save_path = f"{args.save_path}/epoch_{epoch}"
        print(f"Saving model and tokenizer in {save_path}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model and Tokenizer is saved in {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
