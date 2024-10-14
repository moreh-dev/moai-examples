import copy
import datetime
import importlib.metadata
import os
import sys
import time
from typing import Any, Mapping, Optional, Union

from accelerate.logging import get_logger
from datasets import load_dataset
from packaging import version
from peft import PeftModel
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from tqdm.auto import tqdm
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TrainerCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available
from transformers.utils import is_peft_available
from trl import SFTTrainer

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

BAICHUAN_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}{{message['content']}}{% endif %}{% if message['role'] == 'user' %}{{'<reserved_106>' + message['content']}}{% endif %}{% if message['role'] == 'assistant' %}{{'<reserved_107>' + message['content']}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<reserved_107>' }}{% endif %}"

KEY = [
    'model_name_or_path', 'dataset_name_or_path', 'epochs', 'train_batch_size',
    'eval_batch_size', 'block_size', 'lr', 'use_lora'
]
LORA_KEY = ['lora_alpha', 'lora_dropout', 'lora_r']


# Compose pad token mask
def create_mask(input_ids, tokenizer):
    pad_token_ids = (tokenizer.pad_token_id if tokenizer.pad_token_id
                     is not None else tokenizer.eos_token_id)
    return (input_ids != pad_token_ids).long()


# Mask pad tokens
def mask_pads(input_ids, attention_mask, ignore_index=-100):
    idx_mask = attention_mask
    labels = copy.deepcopy(input_ids)
    labels[~idx_mask.bool()] = ignore_index
    return labels


def load_model(args):
    print(f"Loading {args.model_name_or_path} Tokenizer...")
    set_mem_usage_correction_ratio(args)
    configs = AutoConfig.from_pretrained(args.model_name_or_path,
                                         trust_remote_code=True)
    if "baichuan" in configs.architectures[0].lower():
        from model.modeling_baichuan import BaichuanForCausalLM
        model = BaichuanForCausalLM.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                  trust_remote_code=True)
    elif "llama" in configs.architectures[0].lower():
        from model.llama.modeling_llama import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                  trust_remote_code=True)
    elif "qwen2" in configs.architectures[0].lower():
        from model.modeling_qwen2 import Qwen2ForCausalLM
        model = Qwen2ForCausalLM.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif "internlm" in configs.architectures[0].lower():
        from model.internlm.modeling_internlm2 import InternLM2ForCausalLM
        model = InternLM2ForCausalLM.from_pretrained(args.model_name_or_path,
                                                     trust_remote_code=True)
        model = convert_qkv_unfused(model)
        print(
            f"[WARNING] InternLM model is testing, the saved model configs are different from original"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                  trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.use_lora:
        from peft import get_peft_model
        from peft import LoraConfig
        if "baichuan" in configs.architectures[0].lower():
            _target_modules = ["W_pack"]
        else:
            _target_modules = ["q_proj", "v_proj"]
        config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        if any(name in configs.architectures[0].lower()
               for name in ["llama", "qwen2", "gemma"]):
            typecast_untrainable_params(model)
    print_trainable_parameters(model)
    return model, tokenizer


def save_model_and_tokenizer(args, model, tokenizer):
    print(f"Saving model and tokenzier in {args.save_path}")
    config = model.config
    if "internlm" in config.architectures[0].lower():
        model.save_pretrained(args.save_path)
        from model.internlm.modeling_internlm2 import InternLM2ForCausalLM
        model = InternLM2ForCausalLM.from_pretrained(args.save_path,
                                                     trust_remote_code=True)
        model = convert_qkv_fused(model)

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"Model and Tokenzier is saved in {args.save_path}")


def prepare_dataset(args):
    if args.dataset_name_or_path == "bitext/Bitext-customer-support-llm-chatbot-training-dataset":
        dataset = {}
        dataset["train"] = load_dataset(
            args.dataset_name_or_path, split="train[95%:]").with_format("torch")
        dataset["validation"] = load_dataset(
            args.dataset_name_or_path, split="train[:5%]").with_format("torch")
    elif args.dataset_name_or_path == "agileloop/izaz-sequence-of-actions-prediction-dataset-llama2-7b-32k":
        dataset = load_dataset(args.dataset_name_or_path).with_format("torch")
        dataset["train"] = load_dataset(
            args.dataset_name_or_path, split="train[:90%]").with_format("torch")
        dataset["validation"] = load_dataset(
            args.dataset_name_or_path,
            split="train[90%:95%]").with_format("torch")
    else:
        dataset = load_dataset(args.dataset_name_or_path).with_format("torch")

    return dataset


def preprocess_dataset(args, dataset, tokenizer):

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

    def preprocess_agileloop(prompt):
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

    if args.dataset_name_or_path == "bitext/Bitext-customer-support-llm-chatbot-training-dataset":
        dataset['train'] = dataset['train'].map(preprocess,
                                                num_proc=8,
                                                load_from_cache_file=True)
        dataset['validation'] = dataset['validation'].map(
            preprocess, num_proc=8, load_from_cache_file=True)
    elif args.dataset_name_or_path == "agileloop/izaz-sequence-of-actions-prediction-dataset-llama2-7b-32k":
        dataset['train'] = dataset['train'].map(preprocess_agileloop,
                                                num_proc=8,
                                                load_from_cache_file=True)
        dataset['validation'] = dataset['validation'].map(
            preprocess_agileloop, num_proc=8, load_from_cache_file=True)
    else:
        dataset = dataset.map(preprocess, num_proc=8, load_from_cache_file=True)

    return dataset


def create_dataloader(args, tokenizer, preprocessor):
    if 'bitext' in args.dataset_name_or_path.lower(
    ) and 'csv' in args.dataset_name_or_path.lower():
        dataset = load_dataset(
            'csv', data_files=args.dataset_name_or_path).with_format("torch")
        if "validation" not in dataset:
            dataset["train"] = load_dataset(
                'csv',
                data_files=args.dataset_name_or_path,
                split="train[:95%]").with_format("torch")
            dataset["validation"] = load_dataset(
                'csv',
                data_files=args.dataset_name_or_path,
                split="train[95%:]").with_format("torch")
    else:
        dataset = load_dataset(args.dataset_name_or_path,
                               args.dataset_config_name).with_format("torch")
        if "validation" not in dataset:
            dataset["train"] = load_dataset(
                args.dataset_name_or_path,
                args.dataset_config_name,
                split="train[:95%]").with_format("torch")
            dataset["validation"] = load_dataset(
                args.dataset_name_or_path,
                args.dataset_config_name,
                split="train[95%:]").with_format("torch")

    # Tokenize and prepare the input prompt
    def preprocess(prompt):
        tokenized = tokenizer(
            preprocessor(prompt),
            padding="max_length",
            truncation=True,
            max_length=args.block_size,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    def collator(batch):
        return {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch])
        }

    # Preprocess dataset
    dataset = dataset.map(preprocess, num_proc=16, load_from_cache_file=True)

    # Create a DataLoader for the training set
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator)

    # Create a DataLoader for the validation set
    eval_dataloader = torch.utils.data.DataLoader(
        dataset["validation"],
        batch_size=args.eval_batch_size,
        collate_fn=collator)

    return train_dataloader, eval_dataloader


def convert_qkv_unfused(model):
    config = model.config
    num_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    hidden_size = config.hidden_size
    num_key_value_groups = num_heads // num_key_value_heads
    head_dim = hidden_size // num_heads
    for name, module in model.named_modules():
        if name.split('.')[-1] != 'attention':
            continue
        wqkv = module.wqkv
        module.q.weight.requires_grad = False
        module.k.weight.requires_grad = False
        module.v.weight.requires_grad = False
        module.q.weight.copy_(
            wqkv.weight.view(
                num_key_value_heads, num_key_value_groups + 2, head_dim,
                hidden_size)[:, :num_key_value_groups, :, :].contiguous().view(
                    num_key_value_heads * num_key_value_groups * head_dim,
                    hidden_size))
        module.k.weight.copy_(
            wqkv.weight.view(num_key_value_heads, num_key_value_groups + 2,
                             head_dim,
                             hidden_size)[:, -2, :, :].contiguous().view(
                                 num_key_value_heads * head_dim, hidden_size))
        module.v.weight.copy_(
            wqkv.weight.view(num_key_value_heads, num_key_value_groups + 2,
                             head_dim,
                             hidden_size)[:, -1, :, :].contiguous().view(
                                 num_key_value_heads * head_dim, hidden_size))
        if config.bias:
            module.q.bias.requires_grad = False
            module.k.bias.requires_grad = False
            module.v.bias.requires_grad = False
            module.q.bias.copy_(
                wqkv.bias.view(
                    num_heads, num_key_value_groups + 2,
                    head_dim)[:, :num_key_value_groups, :].contiguous().view(
                        num_heads * num_key_value_groups * head_dim))
            module.k.bias.copy_(
                wqkv.bias.view(num_heads, num_key_value_groups + 2,
                               head_dim)[:, -2, :].contiguous().view(num_heads *
                                                                     head_dim))
            module.v.bias.copy_(
                wqkv.bias.view(num_heads, num_key_value_groups + 2,
                               head_dim)[:, -1, :].contiguous().view(num_heads *
                                                                     head_dim))
        del module.wqkv
    return model


def convert_qkv_fused(model):
    config = model.config
    num_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    hidden_size = config.hidden_size
    num_key_value_groups = num_heads // num_key_value_heads
    head_dim = hidden_size // num_heads
    for name, module in model.named_modules():
        if name.split('.')[-1] != 'attention':
            continue
        wqkv = module.wqkv
        q = module.q
        k = module.k
        v = module.v
        wqkv.weight.requires_grad = False
        wqkv.weight.view(num_key_value_heads, num_key_value_groups + 2,
                         head_dim,
                         hidden_size)[:, :num_key_value_groups, :, :].copy_(
                             q.weight.view(num_key_value_heads,
                                           num_key_value_groups, head_dim,
                                           hidden_size))
        wqkv.weight.view(num_key_value_heads, num_key_value_groups + 2,
                         head_dim, hidden_size)[:, -2, :, :].copy_(
                             k.weight.view(num_key_value_heads, head_dim,
                                           hidden_size))
        wqkv.weight.view(num_key_value_heads, num_key_value_groups + 2,
                         head_dim, hidden_size)[:, -1, :, :].copy_(
                             v.weight.view(num_key_value_heads, head_dim,
                                           hidden_size))
        if config.bias:
            wqkv.bias.requires_grad = False
            wqkv.bias.view(num_key_value_heads, num_key_value_groups + 2,
                           head_dim)[:, :num_key_value_groups, :].copy(
                               q.bias.view(num_key_value_heads,
                                           num_key_value_groups, head_dim))
            wqkv.bias.view(num_key_value_heads, num_key_value_groups + 2,
                           head_dim)[:, -2, :].copy(
                               k.bias.view(num_key_value_heads, head_dim))
            wqkv.bias.view(num_key_value_heads, num_key_value_groups + 2,
                           head_dim)[:, -1, :].copy(
                               v.bias.view(num_key_value_heads, head_dim))
        del module.q
        del module.k
        del module.v
    return model


def print_perf(tco_perf_dict):
    # Calculate the averages
    avg_tps = sum(tco_perf_dict["tps"]) / len(
        tco_perf_dict["tps"]) if tco_perf_dict["tps"] else 0
    avg_time_per_20_epoch = sum(tco_perf_dict["time_per_20_epoch"]) / len(
        tco_perf_dict["time_per_20_epoch"]
    ) if tco_perf_dict["time_per_20_epoch"] else 0
    avg_time_per_1_step = sum(tco_perf_dict["time_per_20_epoch"]) / (
        len(tco_perf_dict["time_per_20_epoch"]) * 20 -
        1) if tco_perf_dict["time_per_20_epoch"] else 0
    total_estimated_time = avg_time_per_1_step * (
        tco_perf_dict["total_global_steps"] -
        1) + tco_perf_dict['warmup_duration'] + tco_perf_dict[
            'total_epochs'] * tco_perf_dict['eval_duration']
    train_duration = tco_perf_dict['total_duration'] - tco_perf_dict[
        'eval_duration'] - tco_perf_dict['warmup_duration']
    # Print the results in a formatted way
    print(f"{'Performance Summary':^40}")
    print("=" * 40)
    print(f"{'Train Duration:':<30} {train_duration:.2f} seconds")
    print(
        f"{'Evaluation Duration:':<30} {tco_perf_dict['eval_duration']:.2f} seconds"
    )
    print(
        f"{'Warmup Duration:':<30} {tco_perf_dict['warmup_duration']:.2f} seconds"
    )
    print(
        f"{'Total Duration:':<30} {tco_perf_dict['total_duration']:.2f} seconds"
    )
    print(
        f"{'Total Estimated Duration :':<30} {str(datetime.timedelta(seconds = total_estimated_time))} for {tco_perf_dict['total_epochs'] } epochs"
    )
    print(f"{'Avg TPS:':<30} {avg_tps:.2f} tps")
    print(f"{'Avg Time per 1 Step:':<30} {avg_time_per_1_step:.2f} seconds")
    print("=" * 40)


def print_config(config):
    print("Configuration")
    for key in KEY:
        print(f"{key} : {getattr(config, key)}")
    if config.use_lora:
        for lora_key in LORA_KEY:
            print(f"{lora_key} : {getattr(config, lora_key)}")


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


def typecast_untrainable_params(model):
    for param in model.parameters():
        if not param.requires_grad:
            param.data = param.data.bfloat16()


def set_mem_usage_correction_ratio(args):
    if hasattr(args, "memory_usage_correction_ratio"):
        moreh_config.set_config(
            "advanced_parallelization_memory_usage_correction_ratio",
            args.memory_usage_correction_ratio)


def doc_to_text(doc):
    inputs = " ".join(doc["code_tokens"]).replace("\n", " ")
    inputs = " ".join(inputs.strip().split())

    return inputs


def doc_to_target(doc):
    targets = " ".join(doc["docstring_tokens"]).replace("\n", "")
    targets = " ".join(targets.strip().split())

    return targets


class TrainCallback(TrainerCallback):

    def __init__(self, batch_size, world_size, warm_up_st, total_steps):
        self.duration_st = None
        self.duration_ed = None
        self.step_st = None
        self.warm_up_st = warm_up_st
        self.warm_up_ed = None
        self.eval_st = None
        self.eval_ed = None
        self.batch_size = batch_size
        self.tps = []
        self.step_tps = 0
        self.elapsed_times = []
        self.total_train_steps = total_steps
        self.world_size = world_size

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()
        self.duration_st = time.time()
        self.accum = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.accum += 1

    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step % args.logging_steps == 0) or (state.global_step
                                                             == 1):
            control.should_log = True
        else:
            control.should_log = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step == 1:
            self.warmup_duration = time.time() - self.start
            self.start = time.time()
            self.accum = 0
        else:
            duration = time.time() - self.start
            tps = (args.max_seq_length * self.batch_size * self.accum *
                   self.world_size) / duration
            if 'loss' in logs:
                loss = logs['loss']
                lr = logs['learning_rate']
                if state.is_local_process_zero:
                    print(
                        f"[Step {state.global_step}] | TPS: {tps:.2f} tokens/sec | Loss: {loss:.6f} | LR : {lr:.8f} | Duration for 1 Step: {duration / self.accum:.2f} sec",
                        flush=True)
                self.tps.append(tps)
                self.elapsed_times.append(duration)
            self.accum = 0
            self.start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.duration_ed = time.time()
        self.eval_st = time.time()

    def on_evaluate(self, args, state, control, **kwargs):
        self.eval_ed = time.time()

    def on_train_end(self, args, state, control, **kwargs):
        train_duration = self.duration_ed - self.duration_st
        warm_up_duration = self.warmup_duration
        if args.do_eval:
            eval_duration = self.eval_ed - self.eval_st
        else:
            eval_duration = 0
        avg_tps = sum(self.tps) / len(self.tps)
        avg_time_per_1_step = sum(self.elapsed_times) / (
            len(self.elapsed_times) * args.logging_steps - 1)
        total_steps = self.total_train_steps
        total_estimated_time = warm_up_duration + avg_time_per_1_step * (
            total_steps -
            1) + warm_up_duration + args.num_train_epochs * eval_duration
        days = total_estimated_time // 86400
        total_estimated_time -= days * 86400
        total_duration = train_duration + warm_up_duration + eval_duration
        print()
        print(f"{'Performance Summary':^40}")
        print("=" * 50)
        print(f"{'Total Duration:':<30} {total_duration:.2f} seconds")
        print(
            f"{'  Model Loading Duration:':<30} {warm_up_duration:.2f} seconds")
        print(f"{'  Train Duration:':<30} {train_duration:.2f} seconds")
        print(f"{'  Evaluation Duration:':<30} {eval_duration:.2f} seconds")
        print(
            f"{'Total Estimated Duration:':<30} {str(datetime.timedelta(days=days, seconds=total_estimated_time))} for {args.num_train_epochs} epochs"
        )
        print(f"{'Avg TPS:':<30} {avg_tps:.2f} tps")
        print("=" * 50)
