import torch

from typing import List, Tuple
from pathlib import Path

from nnexp.experiment import Experiment, ExpRun
import nnexp.checkpoint_util as cputil

from .base import TestBase
from nnexp.logging.checkpoint import CheckpointLogger

class TestResume(TestBase):
    def test_load_run(self):
        # setup
        exp_to_save = self.create_dumb_exp(label="foo", one="11", two="22")
        exp_to_save.nepochs = 100
        exp_to_save.get_run().max_epochs = 100
        
        # execute
        _md_path, ckpt_path = self.save_checkpoint(exp_to_save)
        res = cputil.list_experiments(runs_dir=self.runs_dir)

        # assert
        assert 1 == len(res)

        exp_loaded = res[0]
        assert exp_loaded.shortcode == exp_to_save.shortcode
        assert len(exp_to_save.id_diff(exp_loaded)) == 0

        assert len(exp_loaded.runs) == 1
        assert exp_loaded.nepochs == 100
        assert exp_loaded.net_class == 'DumbNet'

        one_run = exp_loaded.get_run()
        assert one_run.checkpoint_path == ckpt_path
        assert one_run.checkpoint_nepochs == 100

    def test_resume(self):
        # setup
        nepochs_orig = 100
        exp_to_save = self.create_dumb_exp(label="foo", one="11", two="22")
        exp_to_save.nepochs = nepochs_orig
        exp_to_save.max_epochs = nepochs_orig
        train_hist = torch.linspace(1.0, 0.1, steps=nepochs_orig).tolist()
        exp_to_save.train_loss_hist = train_hist
        exp_to_save.val_loss_hist = [(epoch, train_hist[epoch]) for epoch in range(0, nepochs_orig, 2)]

        print("runs before save:")
        for i, run in enumerate(exp_to_save.runs):
            cp_path_name = run.checkpoint_path.name if run.checkpoint_path else "<none>"
            print(f"{i}. max_epochs = {run.max_epochs}, cp_nepochs = {run.checkpoint_nepochs}, cp_path = {cp_path_name}")

        # execute.
        # this is an equivalent experiment, but without runs or data. just max_epochs set 
        # and the necessary ID fields to make it associate old and new.
        _md_path, ckpt_path = self.save_checkpoint(exp_to_save)

        print("runs after save:")
        for i, run in enumerate(exp_to_save.runs):
            cp_path_name = run.checkpoint_path.name if run.checkpoint_path else "<none>"
            print(f"{i}. max_epochs = {run.max_epochs}, cp_nepochs = {run.checkpoint_nepochs}, cp_path = {cp_path_name}")

        new_exp = self.create_dumb_exp(label="foo", one="11", two="22")
        new_exp.nepochs = 0

        nepochs_new = nepochs_orig + 10
        res_exps = \
            cputil.resume_experiments(runs_dir=self.runs_dir, 
                                      exps_in=[new_exp], 
                                      max_epochs=nepochs_new)

        # -- assert --
        assert len(res_exps) == 1

        exp_loaded = res_exps[0]
        if exp_to_save.shortcode != exp_loaded.shortcode:
            diff_fields, _save_vals, _other_vals = zip(*exp_to_save.id_diff(exp_loaded))
            print(f"test_resume DIFFS:", " ".join(diff_fields))
        assert exp_loaded.shortcode == exp_to_save.shortcode
        assert exp_loaded.nepochs == 101

        assert len(exp_loaded.runs) == 2

        print("loaded runs:")
        for i, run in enumerate(exp_loaded.runs):
            cp_path_name = run.checkpoint_path.name if run.checkpoint_path else "<none>"
            print(f"{i}. max_epochs = {run.max_epochs}, cp_nepochs = {run.checkpoint_nepochs}, cp_path = {cp_path_name}")

        run_orig = exp_loaded.runs[0]
        assert run_orig.max_epochs == nepochs_orig
        assert run_orig.checkpoint_nepochs == nepochs_orig
        assert run_orig.checkpoint_path == ckpt_path

        run_new = exp_loaded.runs[1]
        assert run_new.max_epochs == nepochs_new
        assert run_new.checkpoint_nepochs == 0
        assert run_new.resumed_from == ckpt_path

class TestCheckpointLogger(TestBase):
    def make_exp_and_logger(self, nepochs: int) -> Tuple[Experiment, CheckpointLogger]:
        exp = self.create_dumb_exp(label="foo")
        exp.max_epochs = nepochs
        logger = CheckpointLogger(basename="test", save_top_k=1, runs_dir=self.runs_dir, update_checkpoint_freq=0)
        logger.on_exp_start(exp)
        return exp, logger

    def make_run(self, nepochs: int, tloss: List[float], vloss: List[float]) -> Experiment:
        exp, logger = self.make_exp_and_logger(nepochs)

        # execute
        for epoch in range(nepochs):
            exp.nepochs = epoch
            exp.train_loss_hist.append(tloss[epoch])
            exp.val_loss_hist.append((epoch, vloss[epoch]))
            logger.on_epoch_end(exp=exp)
        
        return exp

    """
    after some number of epochs have been executed, validate:
    * files:
      * metadata.json exists
      * the number of checkpoint files is len(epochs)
      * those checkpoint files have names starting with epoch_{epoch}
    * runs: for both the in-memory Experiment and metadata on disk:
      * the number of runs is len(epochs)
      * those runs have checkpath_path == (path from 'files' step)
      * those runs have cp_nepochs == {epoch}
    """    
    def validate_checkpoints(self, exp: Experiment, 
                             train_epochs: List[int],
                             val_epochs: List[int]):
        cps_dir = self.checkpoints_dir(exp)

        md_path = Path(cps_dir, "metadata.json")
        assert bool(md_path.exists())

        cp_paths_all = sorted([path for path in Path(cps_dir).iterdir() if path.name.endswith(".ckpt")])
        cp_paths_train = [cp_path for cp_path in cp_paths_all if 'tloss' in cp_path.name]
        cp_paths_val = [cp_path for cp_path in cp_paths_all if 'vloss' in cp_path.name]

        assert len(cp_paths_train) == len(train_epochs)
        assert len(cp_paths_val) == len(val_epochs)

        loaded = cputil.load_from_metadata(md_path=md_path)

        all_epochs_expected = sorted([*train_epochs, *val_epochs])
        all_epochs_loaded = [run.checkpoint_nepochs for run in loaded.runs]
        assert all_epochs_loaded == all_epochs_expected

        runs_exp_train = [run for run in exp.runs if 'tloss' in run.checkpoint_path.name]
        runs_loaded_train = [run for run in loaded.runs if 'tloss' in run.checkpoint_path.name]

        runs_exp_val = [run for run in exp.runs if 'vloss' in run.checkpoint_path.name]
        runs_loaded_val = [run for run in loaded.runs if 'vloss' in run.checkpoint_path.name]
        
        try:
            combos = [[train_epochs, runs_exp_train, runs_loaded_train, cp_paths_train],
                      [val_epochs, runs_exp_val, runs_loaded_val, cp_paths_val]]
            for epochs, runs_exp, runs_loaded, cp_paths in combos:
                for epoch, run_exp, run_loaded, cp_path in zip(epochs, runs_exp, runs_loaded, cp_paths):
                    expected_start = f"epoch_{epoch:04}"
                    assert bool(cp_path.name.startswith(expected_start))
                    assert bool(cp_path.exists())

                    assert run_exp.checkpoint_path == cp_path
                    assert run_exp.checkpoint_nepochs == epoch

                    assert run_loaded.checkpoint_path == cp_path
                    assert run_loaded.checkpoint_nepochs == epoch

        except Exception as e:
            cp_path_names = ", ".join([run.checkpoint_path.name if run.checkpoint_path else "<none>" for run in exp.runs])
            cp_epochs = ", ".join([str(run.checkpoint_nepochs) for run in exp.runs])
            in_epochs = ", ".join(map(str, epochs))
            print("validate_checkpoints failure:")
            print(f"    input epochs = {in_epochs}")
            print(f"       cp_epochs = {cp_epochs}")
            print(f"   cp_path_names = {cp_path_names}")

            raise e

    def test_writes_one_checkpoint(self):
        nepochs = 1
        exp = self.make_run(nepochs, [1.0], [1.0])

        # validate
        self.validate_checkpoints(exp, [0], [0])

    def test_writes_checkpoints_2epochs(self):
        nepochs = 2
        tloss = [2, 1]  # 1 is saved
        vloss = [1, 2]  # 0 is saved
        exp = self.make_run(nepochs, tloss=tloss, vloss=vloss)

        # validate
        self.validate_checkpoints(exp, [1], [0])

    def test_writes_checkpoints_10epochs(self):
        # setup & execute
        nepochs = 10
        tloss = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 0 is saved
        vloss = [5, 4, 3, 2, 1, 2, 3, 4, 5, 6]   # 4 is saved
        exp = self.make_run(nepochs, tloss=tloss, vloss=vloss)

        # validate
        self.validate_checkpoints(exp, [0], [4])

    def test_writes_checkpoints_bothupdate(self):
        # setup
        nepochs = 10
        exp, logger = self.make_exp_and_logger(nepochs)

        # 0 saved,
        # 1 replaces 0
        # 4 replaces 1
        # --
        # 7 replaces 3
        # 9 replaces 7
        tloss = [5, 4, 4, 4, 3] + [3, 3, 2, 2, 1]

        # 0 saved,
        # 2 replaces 0,
        # 4 replaces 2,
        # --
        # 8 replaces 3,
        # 9 replaces 8
        vloss = [5, 5, 4, 4, 3] + [3, 3, 3, 2, 1]

        # execute first batch
        nepochs_half = nepochs // 2
        for epoch in range(0, nepochs_half):
            exp.nepochs = epoch
            exp.train_loss_hist.append(tloss[epoch])
            exp.val_loss_hist.append((epoch, vloss[epoch]))
            logger.on_epoch_end(exp=exp)
        
        self.validate_checkpoints(exp, [4], [4])

        # second batch
        for epoch in range(nepochs_half, nepochs):
            exp.nepochs = epoch
            exp.train_loss_hist.append(tloss[epoch])
            exp.val_loss_hist.append((epoch, vloss[epoch]))
            logger.on_epoch_end(exp=exp)
        
        self.validate_checkpoints(exp, [9], [9])

    def test_resume_creates_run(self):
        # setup
        nepochs = 10
        exp, logger = self.make_exp_and_logger(nepochs)

        tloss = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 0 is saved
        vloss = [5, 4, 3, 2, 1, 2, 3, 4, 5, 6]   # 4 is saved

        exp = self.make_run(nepochs, tloss=tloss, vloss=vloss)

        # validate
        self.validate_checkpoints(exp, [0], [4])

        # empty_exp will match the saved one.
        empty_exp = self.create_dumb_exp(label="foo")
        empty_exp.max_epochs = nepochs
        
        # max_epochs 10 should be replaced by 20.
        resumed_exps = cputil.resume_experiments(exps_in=[empty_exp], max_epochs=20, use_best='tloss', runs_dir=self.runs_dir)
        assert len(resumed_exps) == 1
        resumed_exp = resumed_exps[0]

        # run 0: run created above. cp_path is set, max_epochs 10.
        # run 1: resume run. cp_path None, resumed_from is set, max_epochs 20
        assert len(resumed_exp.runs) == 2
        old_run, new_run = resumed_exp.runs

        assert old_run.max_epochs == 10
        assert old_run.checkpoint_nepochs == 0

        assert new_run.max_epochs == 20
        assert new_run.checkpoint_nepochs == 0
        
    def test_resume_best_tloss(self):
        # setup
        nepochs = 10
        exp, logger = self.make_exp_and_logger(nepochs)

        tloss = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 0 is saved
        vloss = [5, 4, 3, 2, 1, 2, 3, 4, 5, 6]   # 4 is saved

        exp = self.make_run(nepochs, tloss=tloss, vloss=vloss)

        # validate
        self.validate_checkpoints(exp, [0], [4])

        # empty_exp will match the saved one.
        empty_exp = self.create_dumb_exp(label="foo")
        empty_exp.max_epochs = nepochs
        
        # max_epochs 10 should be replaced by 20.
        resumed_exps = cputil.resume_experiments(exps_in=[empty_exp], max_epochs=20, use_best='tloss', runs_dir=self.runs_dir)
        resumed_exp = resumed_exps[0]
        new_run = resumed_exp.get_run()

        # resumed experiment should be ready to start epoch (checkpoint epoch + 1).
        # checkpoint_nepochs should be 0, as it had the best tloss.
        assert resumed_exp.nepochs == 1

        # run 1: resume run. cp_path None, resumed_from is set, max_epochs 20
        assert new_run.checkpoint_nepochs == 0
        assert new_run.checkpoint_path is None
        assert new_run.resumed_from is not None

    def test_resume_best_vloss(self):
        # setup
        nepochs = 10
        exp, logger = self.make_exp_and_logger(nepochs)

        tloss = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 0 is saved
        vloss = [5, 4, 3, 2, 1, 2, 3, 4, 5, 6]   # 4 is saved

        exp = self.make_run(nepochs, tloss=tloss, vloss=vloss)

        # validate
        self.validate_checkpoints(exp, [0], [4])

        # empty_exp will match the saved one.
        empty_exp = self.create_dumb_exp(label="foo")
        empty_exp.max_epochs = nepochs
        
        # max_epochs 10 should be replaced by 20.
        resumed_exps = cputil.resume_experiments(exps_in=[empty_exp], max_epochs=20, use_best='vloss', runs_dir=self.runs_dir)
        resumed_exp = resumed_exps[0]
        new_run = resumed_exp.get_run()

        # resumed experiment should be ready to start epoch (checkpoint epoch + 1).
        # checkpoint_nepochs should be 4, as it had the best tloss.
        assert resumed_exp.nepochs == 5

        # run 0: run created above. cp_path is set, max_epochs 10.
        # run 1: resume run. cp_path None, resumed_from is set, max_epochs 20
        assert new_run.checkpoint_nepochs == 0
        assert new_run.checkpoint_path is None
        assert new_run.resumed_from is not None

        
