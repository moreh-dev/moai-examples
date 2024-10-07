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
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from model.internlm.modeling_internlm2 import InternLM2ForCausalLM

from moreh.driver.common import config as moreh_config


# Compose pad token mask
def create_mask(input_ids, tokenizer):
    pad_token_ids = (tokenizer.pad_token_id if tokenizer.pad_token_id
                     is not None else tokenizer.eos_token_id)
    return (input_ids != pad_token_ids).long()


# Mask pad tokens
def mask_pads(inputs, tokenizer, ignore_index=-100):
    idx_mask = create_mask(inputs, tokenizer)
    labels = copy.deepcopy(inputs)
    labels[~idx_mask.bool()] = ignore_index
    return labels


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# Arguments
def parse_args():
    parser = ArgumentParser(description="LLaMA3 FineTuning")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="meta-llama/Meta-Llama-3-70B",
        help="model name or path",
    )
    parser.add_argument("--epochs",
                        type=int,
                        default=1,
                        help="num training epochs")
    parser.add_argument("--batch-size",
                        type=int,
                        default=512,
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
        default="cnn_dailymail",
        help="dataset name or path",
    )
    parser.add_argument("--lr",
                        type=float,
                        default=0.00001,
                        help="learning rate")
    parser.add_argument("--log-interval",
                        type=int,
                        default=10,
                        help="log interval")
    parser.add_argument(
        "--eval-step",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./llama3_70b_summarization",
        help="model save path",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
    )
    parser.add_argument(
        "--num-micro-batches",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=48,
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
            # if e_step > 10:
            #     break
            e_input_ids = e_batch
            e_inputs, e_labels = e_input_ids, mask_pads(e_input_ids, tokenizer)
            e_attn_mask = create_mask(e_inputs, tokenizer)
            e_position_ids = e_attn_mask.long().cumsum(-1) - 1
            e_position_ids.masked_fill_(e_attn_mask == 0, 1)
            #e_position_ids = e_position_ids.cuda()
            if e_step % 10 == 0:
                logger.info(f"EVAL STEP: {e_step} / {len(eval_dataloader)}")
            e_outputs = model(
                e_inputs.cuda(),
                #attention_mask=e_attn_mask.cuda(),
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
    # tokenizer.pad_token_id = 0
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
    if args.dataset_name_or_path == "cnn_dailymail":
        dataset = load_dataset(args.dataset_name_or_path,
                               "3.0.0").with_format("torch")
        dataset["train"] = load_dataset(args.dataset_name_or_path,
                                        "3.0.0",
                                        split="train[:1%]").with_format("torch")
        dataset["validation"] = load_dataset(
            args.dataset_name_or_path, "3.0.0",
            split="validation[:1%]").with_format("torch")
        dataset["test"] = load_dataset(args.dataset_name_or_path,
                                       "3.0.0",
                                       split="test[:1%]").with_format("torch")
    else:
        dataset = load_dataset(args.dataset_name_or_path).with_format("torch")
        if "validation" not in dataset:
            dataset["train"] = load_dataset(
                args.dataset_name_or_path,
                split="train[:5%]").with_format("torch")
            dataset["validation"] = load_dataset(
                args.dataset_name_or_path,
                split="train[95%:]").with_format("torch")
    # Construct a formatted prompt
    def create_prompt(prompt):
        full_prompt = (
            f"[SUMMARIZE] {prompt['article']} [/SUMMARIZE]\n{prompt['highlights']}</s>"
        )
        return full_prompt

    # Tokenize and prepare the input prompt
    def preprocess(prompt):
        input_ids = tokenizer(
            create_prompt(prompt),
            return_attention_mask=False,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            max_length=args.block_size,
            return_tensors='pt'
        )["input_ids"]
        return {"input_ids": input_ids}

    print("Preprocessing dataset...")

    # Preprocess dataset

    def preprocess_general(prompt):
        chat = [{
            "role": "user",
            "content": f"{prompt['Instruction']}"
        }, {
            "role": "assistant",
            "content": f"{prompt['Response']}"
        }]
        
        #chat = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False, padding="max_length", max_length=args.block_size)
        chat = tokenizer.apply_chat_template(chat, tokenize=False)
        result = tokenizer(chat,
                           truncation=True,
                           max_length=args.block_size,
                           padding="max_length")
        result['labels'] = copy.deepcopy(result['input_ids'])
        ret = {}
        ret['formatted_chat'] = result['input_ids']
        return ret

    if args.dataset_name_or_path == "cnn_dailymail":
        dataset = dataset.map(preprocess,
                              num_proc=1,
                              load_from_cache_file=True)
    else:
        dataset = dataset.map(preprocess_general,
                              num_proc=1)
    
    # Create a DataLoader for the training set
    # Use collate_fn to ensure that all data samples are of the sample length
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn = lambda batch: torch.stack([torch.tensor(sample['formatted_chat']) for sample in batch])
    )

    # Use collate_fn to ensure that all data samples are of the sample length
    # Create a DataLoader for the validation set
    eval_dataloader = torch.utils.data.DataLoader(
        dataset["validation"],
        batch_size=args.eval_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn = lambda batch: torch.stack([torch.tensor(sample['formatted_chat']) for sample in batch])
    )

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
            # if step > 10: break
            start_time = time.perf_counter()
            #print(batch)
            input_ids = batch # Because of collate_fn, just use batch directly.
            print(input_ids.shape)
            inputs, labels = input_ids, mask_pads(input_ids, tokenizer)
            attn_mask = create_mask(inputs, tokenizer)
            position_ids = attn_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attn_mask == 0, 1)
            #position_ids = position_ids.cuda()
            outputs = model(
                input_ids.cuda(),
                #attention_mask=attn_mask.cuda(),
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
        # eval(model, eval_dataloader, tokenizer)
        model.train()
        st = time.time()
    print("Training Done")
    print("Saving Model...")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"Model saved in {args.save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
