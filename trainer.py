from typing import Iterable
from pathlib import Path
import datetime
import gc

import torch, torch.optim
from experiment import Experiment

class TrainerLogger:
    basename: str
    started_at: datetime.datetime
    started_at_str: str
    runs_dir: Path

    def __init__(self, basename: str, started_at: datetime.datetime = None, runs_dir: Path = None):
        if started_at is None:
            started_at = datetime.datetime.now()

        self.basename = basename
        self.runs_dir = runs_dir or Path("runs")
        self.started_at = started_at
        self.started_at_str = self.started_at.strftime("%Y%m%d-%H%M%S")
    
    """
    get filename for experiment with no subdirs
    """
    def get_exp_base(self, exp: Experiment) -> str:
        return f"{exp.created_at_short}-{exp.shortcode}--{exp.label}"

    """
    runs/{subdir}/{basename}/{exp_base}
    """
    def get_exp_path(self, subdir: str, exp: Experiment, mkdir: bool = False) -> Path:
        exp_name = self.get_exp_base(exp)

        path = Path(self.runs_dir, f"{subdir}-{self.basename}", exp_name)
        if mkdir:
            path.mkdir(exist_ok=True, parents=True)
        return path

    def get_run_dir(self, subdir: str, include_timestamp: bool = True) -> Path:
        path = Path(self.runs_dir, f"{subdir}-{self.basename}")
        if include_timestamp:
            path = Path(path, self.started_at_str)
        path.mkdir(exist_ok=True, parents=True)
        return path

    def on_exp_start(self, exp: Experiment):
        pass

    def on_exp_end(self, exp: Experiment):
        pass

    """
         epoch: current epoch
         batch: batch num of current epoch
    batch_size: the length of the current batch (may be less than exp.batch_size on the last batch)
    train_loss: training loss for this batch only
    """
    def on_batch(self, exp: Experiment, batch: int, batch_size: int, train_loss_batch: float):
        pass

    """
    gets called at the end of Trainer.on_epoch_end

         epoch: current (just-ended) epoch
    """
    def on_epoch_end(self, exp: Experiment):
        pass

    """
    gets called every trainer.update_frequency.

         epoch: current epoch
         batch: current batch in epoch
    batch_size: the length of the current batch (may be less than exp.batch_size on the last batch)
    train_loss: training loss for epoch so far
    """
    def print_status(self, exp: Experiment, batch: int, batch_size: int, train_loss_epoch: float):
        pass

    """
    gets called (trainer.desired_val_count) times per Experiment.
    val_loss has already been computed.
    """
    def update_val_loss(self, exp: Experiment):
        pass

"""
At the end of training an epoch:
- Trainer.on_epoch_end
  - (possibly) Logger.update_val_loss
  - Logger.on_epoch_end

At the end of training an experiment:
- everything in end of epoch training
- Trainer.on_exp_end
  - logger.on_exp_end
  - exp.end
"""
class Trainer:
    experiments: Iterable[Experiment]
    nexperiments: int
    logger: TrainerLogger

    total_samples = 0    # samples trained so far
    total_batches = 0    # batches (steps) trained so far
    total_epochs = 0     # epochs trained so far

    last_epoch_started_at: datetime.datetime = None
    last_print: datetime.datetime = None

    last_val_at: datetime.datetime = None
    val_limit_frequency: datetime.timedelta = None

    update_frequency: datetime.timedelta

    # TODO: remove 'nexperiments' as Experiments can lazy load now.
    def __init__(self, 
                 experiments: Iterable[Experiment], nexperiments: int,
                 update_frequency = 10, val_limit_frequency = 0,
                 logger: TrainerLogger = None):
        self.experiments = experiments
        self.nexperiments = nexperiments
        self.logger = logger
        self.update_frequency = datetime.timedelta(seconds=update_frequency)
        self.val_limit_frequency = datetime.timedelta(seconds=val_limit_frequency)

    def on_exp_start(self, exp: Experiment, exp_idx: int):
        exp.start(exp_idx)
        if self.logger is not None:
            self.logger.on_exp_start(exp)

        if self.val_limit_frequency:
            self.last_val_at = datetime.datetime.now()

        self.nbatches_per_epoch = len(exp.train_dataloader)

        # track how many samples the exp started with, in case it was resumed.
        self.exp_start_nsamples = exp.nsamples
        self.exp_start_nepochs = exp.nepochs

    def on_exp_end(self, exp: Experiment):
        exp.end()
        if self.logger is not None:
            self.logger.on_exp_end(exp)

        # TODO: still might be leaking memory. not sure yet. TODO: run more systematic tests.
        exp.end_cleanup()
        gc.collect()
        torch.cuda.empty_cache()
    
    def print_status(self, exp: Experiment, batch: int, batch_size: int, train_loss_epoch: float):
        now = datetime.datetime.now()
        if ((now - self.last_print) >= self.update_frequency or
             (batch == batch_size - 1 and exp.nepochs == exp.max_epochs - 1)):
            
            timediff = (now - exp.started_at)

            # compute per/sec since the beginning of this experiment.
            samples_diff = exp.nsamples - self.exp_start_nsamples
            samples_per_sec = samples_diff / timediff.total_seconds()
            batch_per_sec = samples_per_sec / batch_size
            epoch_per_sec = batch_per_sec / self.nbatches_per_epoch

            if epoch_per_sec < 1:
                sec_per_epoch = 1 / epoch_per_sec
                epoch_rate = f"{sec_per_epoch:.3f}s/epoch"
            else:
                epoch_rate = f"epoch/s {epoch_per_sec:.3f}"

            print(f"epoch {exp.nepochs+1}/{exp.max_epochs} | batch {batch+1}/{self.nbatches_per_epoch} | \033[1;32mtrain loss {train_loss_epoch:.5f}\033[0m | samp/s {samples_per_sec:.3f} | {epoch_rate} | exp {exp.shortcode}")

            self.last_print = now
            self.last_print_total_samples = self.total_samples

            if self.logger is not None:
                self.logger.print_status(exp, batch, batch_size, train_loss_epoch)

    # override this for new behavior after each epoch.
    def on_epoch_end(self, exp: Experiment, device: str):
        # figure out validation loss
        did_val = False
        now = datetime.datetime.now()
        if not self.val_limit_frequency or (now - self.last_val_at) >= self.val_limit_frequency:
            did_val = True
            if self.val_limit_frequency:
                self.last_val_at = now

            val_start = now
            with torch.no_grad():
                exp.net.eval()
                exp_batch = 0
                val_loss = 0.0
                for batch, one_tuple in enumerate(exp.val_dataloader):
                    inputs = one_tuple[0]
                    if not isinstance(inputs, list):
                        inputs = [inputs]
                    truth = one_tuple[1]

                    inputs = [inp.to(device) for inp in inputs]
                    truth = truth.to(device)

                    exp_batch += 1
                    val_out = exp.net(*inputs)
                    loss = exp.loss_fn(val_out, truth)

                    if loss.isnan():
                        print(f"!! validation loss {loss} at epoch {exp.nepochs + 1}, batch {batch} -- returning!")
                        return False

                    val_loss += loss.item()

            val_end = datetime.datetime.now()

            val_loss /= exp_batch
            exp.val_loss_hist.append((exp.nepochs, val_loss))

            train_elapsed = (val_start - self.last_epoch_started_at).total_seconds()
            val_elapsed = (val_end - val_start).total_seconds()
            
            exp_elapsed = (val_end - exp.started_at).total_seconds()
            exp_elapsed_min = int(exp_elapsed / 60)
            exp_elapsed_sec = int(exp_elapsed) % 60

            epochs_since_start = (exp.nepochs - self.exp_start_nepochs)
            exp_expected = exp_elapsed * exp.max_epochs / (epochs_since_start + 1)
            exp_expected_min = int(exp_expected / 60)
            exp_expected_sec = int(exp_expected) % 60

            print(f"epoch {exp.nepochs + 1}/{exp.max_epochs} "
                  f"| \033[1;32mtrain loss {exp.last_train_loss:.5f}, val loss {val_loss:.5f}\033[0m "
                  f"| train {train_elapsed:.2f}s, val {val_elapsed:.2f}s "
                  f"| exp {exp.exp_idx+1}/{self.nexperiments}: {exp_elapsed_min}m{exp_elapsed_sec}s / {exp_expected_min}m{exp_expected_sec}s")
            if self.logger is not None:
                self.logger.update_val_loss(exp)

        if self.logger is not None:
            self.logger.on_epoch_end(exp)

        if did_val:
            print()

        # increase nepochs.
        exp.nepochs += 1

    def train(self, device = "cpu", use_amp = False):
        self.last_print = datetime.datetime.now()
        for exp_idx, exp in enumerate(self.experiments):
            # reset it between runs.
            if use_amp:
                self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
            else:
                self.scaler = None

            self.on_exp_start(exp, exp_idx)
            if exp.skip:
                print(f"skip {exp_idx + 1}/{self.nexperiments} | {exp.label}")
                continue

            print()
            print(f"\033[1mtrain {exp_idx+1}/{self.nexperiments}: exp {exp.shortcode} | {exp.nparams() / 1e6:.3f}M params | {exp.label}\033[0m")
            if exp.nepochs > 0:
                print(f"* \033[1;32mresuming from {exp.nepochs} epochs\033[0m")
            
            for _epoch in range(exp.nepochs, exp.max_epochs):
                stepres = self.train_epoch(exp, device=device)
                if not stepres:
                    # something went wrong in that step. 
                    break
            
            exp.ended_at = datetime.datetime.now()

            self.on_exp_end(exp)

    def train_epoch(self, exp: Experiment, device: str) -> bool:
        run = exp.get_run()
        grad_accum = run.grad_accum or 1

        self.total_epochs += 1
        self.last_epoch_started_at = datetime.datetime.now()

        total_loss = 0.0
        for batch, one_tuple in enumerate(exp.train_dataloader):
            inputs = one_tuple[0]
            if not isinstance(inputs, list):
                inputs = [inputs]
            truth = one_tuple[1]
            # there might be other stuff, which we'll ignore.

            batch_size = len(inputs[0])

            self.total_batches += 1
            self.total_samples += batch_size
            exp.nsamples += batch_size

            inputs = [inp.to(device) for inp in inputs]
            truth = truth.to(device)

            exp.net.train()
            if self.scaler is not None:
                with torch.cuda.amp.autocast_mode.autocast():
                    out = exp.net(*inputs)
                    loss = exp.loss_fn(out, truth)
            else:
                out = exp.net(*inputs)
                loss = exp.loss_fn(out, truth)

            if loss.isnan():
                # not sure if there's a way out of this...
                print(f"!! train loss {loss} at epoch {exp.nepochs + 1}, batch {batch} -- returning!")
                return False

            if batch % grad_accum == 0 or batch == len(exp.train_dataloader) - 1:
                if self.scaler is not None:
                    loss_scaled = self.scaler.scale(loss)
                    loss_scaled.backward()
                else:
                    loss.backward()
                total_loss += loss.item()

                if self.scaler is not None:
                    self.scaler.step(exp.optim)
                    self.scaler.update()
                else:
                    exp.optim.step()
                exp.optim.zero_grad(set_to_none=True)
            exp.net.eval()

            exp.nbatches += 1

            if self.logger is not None:
                self.logger.on_batch(exp, batch, batch_size, loss.item())
            self.print_status(exp, batch, batch_size, total_loss / (batch + 1))

        exp.sched.step()

        total_loss = total_loss / (batch + 1)
        exp.train_loss_hist.append(total_loss)
        exp.lr_hist.append(exp.cur_lr)

        self.on_epoch_end(exp, device=device)

        return True

