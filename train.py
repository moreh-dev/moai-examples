import argparse
import copy
import time

from accelerate import Accelerator
from accelerate.logging import get_logger
import datasets
from datasets import load_dataset
from peft import get_peft_model
from peft import LoraConfig
import torch
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from tqdm.auto import tqdm
import transformers
from transformers import AutoTokenizer
from trl import SFTConfig
from trl import SFTTrainer

from utils import *


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="/root/poc/pretrained_models/Meta-Llama-3-70B-Instruct")
    parser.add_argument(
        "--dataset-name-or-path",
        type=str,
        default="bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    parser.add_argument("--block-size", type=int, default=32768)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir",
                        type=str,
                        default="/root/poc/checkpoints/llama_lora_finetuned")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--log-interval", type=int, default=5)
    args = parser.parse_args()
    return args


def main(args):
    torch.moreh.option.enable_advanced_parallelization()

    accelerator = Accelerator()
    world_size = accelerator.num_processes
    logger = get_logger(__name__)
    logger.info(accelerator.state, main_process_only=True)
    logger.warning(accelerator.state, main_process_only=True)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    model, tokenizer = load_model(args)

    if args.dataset_name_or_path == "bitext/Bitext-customer-support-llm-chatbot-training-dataset":
        dataset = {}
        dataset["train"] = load_dataset(args.dataset_name_or_path,
                                        split="train[95%:]")
        dataset["validation"] = load_dataset(args.dataset_name_or_path,
                                             split="train[:5%]")
    else:
        dataset = load_dataset(args.dataset_name_or_path)

    def preprocess(prompt):
        chat = [
            {
                "role": "user",
                "content": f"{prompt['instruction']}"
            },
            {
                "role": "assistant",
                "content": f"{prompt['response']}"
            },
        ]
        chat = tokenizer.apply_chat_template(chat, tokenize=False)
        result = tokenizer(chat,
                           truncation=True,
                           max_length=args.block_size,
                           padding="max_length")
        result['labels'] = copy.deepcopy(result['input_ids'])
        return result

    if args.dataset_name_or_path == "bitext/Bitext-customer-support-llm-chatbot-training-dataset":
        dataset['train'] = dataset['train'].map(preprocess,
                                                num_proc=8,
                                                load_from_cache_file=True)
        dataset['validation'] = dataset['validation'].map(
            preprocess, num_proc=8, load_from_cache_file=True)
    else:
        dataset = dataset.map(preprocess, num_proc=8, load_from_cache_file=True)

    def collator(batch):
        return {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch])
        }

    # SFTConfig
    trainer_config = SFTConfig(
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        output_dir=args.output_dir,
        max_seq_length=1024,
        optim='adamw_torch',
        lr_scheduler_type="cosine",
        learning_rate=args.lr,
        warmup_steps=50,
        bf16=True,
        do_eval=True,
        eval_strategy="epoch",
        logging_steps=args.log_interval,
        report_to='none',
        logging_nan_inf_filter=False,
        save_strategy="no",
        max_grad_norm=0,
    )

    warm_up_st = time.time()

    total_train_steps = (len(dataset["train"]) //
                         (world_size * args.train_batch_size)) * args.num_epochs

    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        args=trainer_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        #data_collator=collator,
        callbacks=[
            TrainCallback(batch_size=args.train_batch_size,
                          world_size=world_size,
                          warm_up_st=warm_up_st,
                          total_steps=total_train_steps)
        ])

    trainer.train()
    if accelerator.is_local_main_process:
        print("Skip to save model")


if __name__ == "__main__":
    args = arg_parse()
    main(args)
