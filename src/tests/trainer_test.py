from typing import List, Tuple, Callable
import datetime
import math

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .base import TestBase
from nnexp.experiment import Experiment
from nnexp.training import trainer
from nnexp import checkpoint_util
from nnexp.logging.checkpoint import CheckpointLogger

NSAMPLES = 100
BATCH_SIZE = 1
NEPOCHS = 10
EXP_NBATCHES = math.ceil(NSAMPLES / BATCH_SIZE) * NEPOCHS
EXP_NSAMPLES = NSAMPLES * NEPOCHS

def create_dataset() -> List[Tuple[Tensor, Tensor]]:
    data = torch.linspace(start=torch.tensor(1.0), end=torch.tensor(0.1), steps=NSAMPLES)
    data.requires_grad_(True)
    return list(zip(*[data, data]))

def create_dataloader() -> DataLoader:
    dataset = create_dataset()
    return DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

def loss_fn(exp: Experiment) -> Callable[[Tensor, Tensor], Tensor]:
    def fn(out: Tensor, _truth: Tensor) -> Tensor:
        return out / (exp.nepochs + 1)
    return fn

class TestCheckpoints(TestBase):
    def list_checkpoints(self) -> List[Experiment]:
        return checkpoint_util.list_experiments(runs_dir=self.runs_dir)

    def create_dumb_exp(self) -> Experiment:
        exp = super().create_dumb_exp(label="test")

        # setup
        train_dl = create_dataloader()
        val_dl = create_dataloader()

        exp.max_epochs = NEPOCHS
        exp.train_dataloader = train_dl
        exp.val_dataloader = val_dl
        exp.loss_fn = loss_fn(exp)

        return exp
    
    def test_train(self):
        exp = self.create_dumb_exp()
        tr = trainer.Trainer(experiments=[exp], nexperiments=1)

        # execute
        tr.train()

        # assert - experiment values
        assert len(exp.train_loss_hist) == NEPOCHS

        assert exp.nepochs == NEPOCHS
        assert exp.nbatches == EXP_NBATCHES
        assert exp.nsamples == EXP_NSAMPLES

    def test_train_checkpoints(self):
        exp = self.create_dumb_exp()

        logger = CheckpointLogger(basename="test", save_top_k=1, runs_dir=self.runs_dir)
        tr = trainer.Trainer(experiments=[exp], nexperiments=1, logger=logger)

        # execute. capture time without microseconds, as timestamps are written to second precision
        before_train = datetime.datetime.now()
        before_train = before_train - datetime.timedelta(microseconds=before_train.microsecond)
        tr.train()

        # assert - runs, and checkpoint values set by CheckpointLogger/checkpoint_util
        # one checkpoint for each of training and validation loss.
        assert len(exp.runs) == 2
        for one_run in exp.runs:
            assert one_run.max_epochs == NEPOCHS
            assert one_run.checkpoint_nepochs == NEPOCHS - 1
            assert one_run.checkpoint_nbatches== EXP_NBATCHES
            assert one_run.checkpoint_nsamples == EXP_NSAMPLES
            assert one_run.checkpoint_path is not None
            assert one_run.checkpoint_at is not None
            assert bool(one_run.checkpoint_at >= before_train)

    def test_load_trained_checkpoints(self):
        start_exp = self.create_dumb_exp()

        logger = CheckpointLogger(basename="test", save_top_k=1, runs_dir=self.runs_dir)
        tr = trainer.Trainer(experiments=[start_exp], nexperiments=1, logger=logger)

        # execute. capture time without microseconds, as timestamps are written to second precision
        before_train = datetime.datetime.now()
        before_train = before_train - datetime.timedelta(microseconds=before_train.microsecond)
        tr.train()

        # assert
        exps = self.list_checkpoints()
        assert len(exps) == 1

        load_exp = exps[0]
        assert load_exp.nepochs == NEPOCHS - 1

        # one run for each of training, validation loss.
        assert len(load_exp.runs) == 2

        for one_run in load_exp.runs:
            assert one_run.max_epochs == NEPOCHS
            assert one_run.checkpoint_nepochs == NEPOCHS - 1
            assert one_run.checkpoint_nbatches== EXP_NBATCHES
            assert one_run.checkpoint_nsamples == EXP_NSAMPLES
            assert one_run.checkpoint_path is not None
            assert one_run.checkpoint_at is not None
            assert bool(one_run.checkpoint_at >= before_train)

    def test_resume_trained_checkpoints(self):
        start_exp = self.create_dumb_exp()

        logger = CheckpointLogger(basename="test", save_top_k=1, runs_dir=self.runs_dir)
        tr = trainer.Trainer(experiments=[start_exp], nexperiments=1, logger=logger)

        # execute. capture time without microseconds, as timestamps are written to second precision
        before_train = datetime.datetime.now()
        before_train = before_train - datetime.timedelta(microseconds=before_train.microsecond)
        tr.train()

        # assert
        new_exp = self.create_dumb_exp()
        resume_exps = checkpoint_util.resume_experiments(exps_in=[new_exp], max_epochs=NEPOCHS * 2, runs_dir=self.runs_dir)


        # assert
        assert len(resume_exps) == 1

        resume_exp = resume_exps[0]
        assert resume_exp.nepochs == NEPOCHS
        assert len(resume_exp.runs) == 3

        # run 0 = training save
        # run 1 = val save
        # run 2 = resume
        for cp_run in resume_exp.runs[:2]:
            assert cp_run.max_epochs == NEPOCHS
            assert cp_run.checkpoint_nepochs == NEPOCHS - 1
            assert cp_run.checkpoint_nbatches== EXP_NBATCHES
            assert cp_run.checkpoint_nsamples == EXP_NSAMPLES
            assert cp_run.checkpoint_path is not None
            assert cp_run.checkpoint_at is not None
            assert bool(cp_run.checkpoint_at >= before_train)

        resume_run = resume_exp.runs[2]
        assert resume_run.max_epochs == NEPOCHS * 2
        assert resume_run.checkpoint_nepochs == 0
        assert resume_run.checkpoint_nbatches == 0
        assert resume_run.checkpoint_nsamples == 0
        assert resume_run.checkpoint_path is None
        assert resume_run.checkpoint_at is None
