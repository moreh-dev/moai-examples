import datetime
import sys
import time

from accelerate.logging import get_logger
import torch
from tqdm.auto import tqdm
from transformers import TrainerCallback


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
