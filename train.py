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

from model.modeling_llama import LlamaForCausalLM
from utils import TrainCallback


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", action="store_true")
    parser.add_argument(
        "--model",
        type=str,
        default="/root/poc/pretrained_models/Meta-Llama-3-70B-Instruct")
    parser.add_argument(
        "--dataset",
        type=str,
        default=
        "/root/poc/datasets/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )
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

    # TODO load_model function
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation='flash_attention_2' if args.flash_attn else "eager",
        use_cache=False,
    )

    # TODO load_tokenizer function
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right",
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = 1024
    tokenizer.padding = "max_length"

    dataset = load_dataset(args.dataset).with_format("torch")
    dataset["train"] = load_dataset(args.dataset, split="train[5%:]")
    dataset["validation"] = load_dataset(args.dataset, split="train[:5%]")

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
                           max_length=1024,
                           padding="max_length")
        result['labels'] = copy.deepcopy(result['input_ids'])
        return result

    dataset = dataset.map(preprocess, num_proc=16)
    dataset = dataset.remove_columns(
        ['flags', 'instruction', 'category', 'intent', 'response'])

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

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    warm_up_st = time.time()
    if args.lora:
        model = get_peft_model(model, peft_config)

    total_train_steps = (len(dataset["train"]) //
                         (world_size * args.train_batch_size)) * args.num_epochs

    trainer = SFTTrainer(model,
                         args=trainer_config,
                         train_dataset=dataset['train'],
                         eval_dataset=dataset['validation'],
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
