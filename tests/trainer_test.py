from typing import List, Tuple, Callable
import datetime
import math

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .base import TestBase
from experiment import Experiment
import trainer
import checkpoint_util
from loggers.checkpoint import CheckpointLogger

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
        self.assertEqual(NEPOCHS, len(exp.train_loss_hist))

        self.assertEqual(NEPOCHS, exp.nepochs)
        self.assertEqual(EXP_NBATCHES, exp.nbatches)
        self.assertEqual(EXP_NSAMPLES, exp.nsamples)

    def test_train_checkpoints(self):
        exp = self.create_dumb_exp()

        logger = CheckpointLogger(basename="test", save_top_k=1, runs_dir=self.runs_dir)
        tr = trainer.Trainer(experiments=[exp], nexperiments=1, logger=logger)

        # execute. capture time without microseconds, as timestamps are written to second precision
        before_train = datetime.datetime.now()
        before_train = before_train - datetime.timedelta(microseconds=before_train.microsecond)
        tr.train()

        # assert - runs, and checkpoint values set by CheckpointLogger/checkpoint_util
        # look at the run
        self.assertEqual(1, len(exp.runs))
        one_run = exp.get_run()
        self.assertEqual(NEPOCHS, one_run.max_epochs)
        self.assertEqual(NEPOCHS, one_run.checkpoint_nepochs)
        self.assertEqual(EXP_NBATCHES, one_run.checkpoint_nbatches)
        self.assertEqual(EXP_NSAMPLES, one_run.checkpoint_nsamples)
        self.assertIsNotNone(one_run.checkpoint_path)
        self.assertIsNotNone(one_run.checkpoint_at)
        self.assertTrue(one_run.checkpoint_at >= before_train)

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
        self.assertEqual(1, len(exps))

        load_exp = exps[0]
        self.assertEqual(NEPOCHS, load_exp.nepochs)

        self.assertEqual(1, len(load_exp.runs))

        one_run = load_exp.get_run()
        self.assertEqual(NEPOCHS, one_run.max_epochs)
        self.assertEqual(NEPOCHS, one_run.checkpoint_nepochs)
        self.assertEqual(EXP_NBATCHES, one_run.checkpoint_nbatches)
        self.assertEqual(EXP_NSAMPLES, one_run.checkpoint_nsamples)
        self.assertIsNotNone(one_run.checkpoint_path)
        self.assertIsNotNone(one_run.checkpoint_at)
        print("test_load_trained_checkpoints:")
        print(f"  {one_run.checkpoint_at=}")
        print(f"           {before_train=}")
        self.assertTrue(one_run.checkpoint_at >= before_train)

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
        self.assertEqual(1, len(resume_exps))

        resume_exp = resume_exps[0]
        self.assertEqual(NEPOCHS, resume_exp.nepochs)
        self.assertEqual(2, len(resume_exp.runs))

        first_run = resume_exp.runs[0]
        self.assertEqual(NEPOCHS, first_run.max_epochs)
        self.assertEqual(NEPOCHS, first_run.checkpoint_nepochs)
        self.assertEqual(EXP_NBATCHES, first_run.checkpoint_nbatches)
        self.assertEqual(EXP_NSAMPLES, first_run.checkpoint_nsamples)
        self.assertIsNotNone(first_run.checkpoint_path)
        self.assertIsNotNone(first_run.checkpoint_at)
        self.assertTrue(first_run.checkpoint_at >= before_train)

        resume_run = resume_exp.runs[1]
        self.assertEqual(NEPOCHS * 2, resume_run.max_epochs)
        self.assertEqual(0, resume_run.checkpoint_nepochs)
        self.assertEqual(0, resume_run.checkpoint_nbatches)
        self.assertEqual(0, resume_run.checkpoint_nsamples)
        self.assertIsNone(resume_run.checkpoint_path)
        self.assertIsNone(resume_run.checkpoint_at)
