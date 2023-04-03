import torch

from typing import Tuple
from experiment import Experiment, ExpRun
from pathlib import Path
import checkpoint_util as cputil

from .base import TestBase
from loggers.checkpoint import CheckpointLogger

class TestResume(TestBase):
    def test_save_creates_run(self):
        # setup
        exp_to_save = self.create_dumb_exp(label="foo", one="11", two="22")
        exp_to_save.nepochs = 100
        self.assertEqual(0, len(exp_to_save.runs))

        # execute
        _md_path, ckpt_path = self.save_checkpoint(exp_to_save)

        # assert
        self.assertEqual(1, len(exp_to_save.runs))
        one_run = exp_to_save.get_run()
        self.assertEqual(ckpt_path, one_run.checkpoint_path)
        self.assertEqual(100, one_run.checkpoint_nepochs)

    def test_load_run(self):
        # setup
        exp_to_save = self.create_dumb_exp(label="foo", one="11", two="22")
        exp_to_save.nepochs = 100
        
        # execute
        _md_path, ckpt_path = self.save_checkpoint(exp_to_save)
        res = cputil.list_experiments(runs_dir=self.runs_dir)

        # assert
        self.assertEqual(1, len(res))

        exp_loaded = res[0]
        self.assertEqual(exp_to_save.shortcode, exp_loaded.shortcode)
        self.assertEqual(0, len(exp_to_save.id_diff(exp_loaded)))

        self.assertEqual(1, len(exp_loaded.runs))
        self.assertEqual(100, exp_loaded.nepochs)
        self.assertEqual('DumbNet', exp_loaded.net_class)

        one_run = exp_loaded.get_run()
        self.assertEqual(ckpt_path, one_run.checkpoint_path)
        self.assertEqual(100, one_run.checkpoint_nepochs)

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
        self.assertEqual(1, len(res_exps))

        exp_loaded = res_exps[0]
        if exp_to_save.shortcode != exp_loaded.shortcode:
            diff_fields, _save_vals, _other_vals = zip(*exp_to_save.id_diff(exp_loaded))
            print(f"test_resume DIFFS:", " ".join(diff_fields))
        self.assertEqual(exp_to_save.shortcode, exp_loaded.shortcode)
        self.assertEqual(100, exp_loaded.nepochs)

        self.assertEqual(2, len(exp_loaded.runs))

        print("loaded runs:")
        for i, run in enumerate(exp_loaded.runs):
            cp_path_name = run.checkpoint_path.name if run.checkpoint_path else "<none>"
            print(f"{i}. max_epochs = {run.max_epochs}, cp_nepochs = {run.checkpoint_nepochs}, cp_path = {cp_path_name}")

        run_orig = exp_loaded.runs[0]
        self.assertEqual(nepochs_orig, run_orig.max_epochs)
        self.assertEqual(nepochs_orig, run_orig.checkpoint_nepochs)
        self.assertEqual(ckpt_path, run_orig.checkpoint_path)

        run_new = exp_loaded.runs[1]
        self.assertEqual(nepochs_new, run_new.max_epochs)
        self.assertEqual(0, run_new.checkpoint_nepochs)
        self.assertEqual(ckpt_path, run_new.resumed_from)

class TestCheckpointLogger(TestBase):
    def setup(self, nepochs: int) -> Experiment:
        if nepochs > 1:
            vals_up = torch.linspace(start=0.1, end=1.0, steps=nepochs // 2)
            vals_down = torch.linspace(start=1.0, end=0.1, steps=nepochs // 2)
            tloss = torch.cat([vals_up, vals_down]).tolist()
            vloss = torch.cat([vals_down, vals_up]).tolist()
        else:
            tloss = [1.0]
            vloss = [1.0]

        # best tloss will be at the end, and best vloss will be in the middle.
        exp = self.create_dumb_exp(label="foo")
        logger = CheckpointLogger(basename="test", save_top_k=1, runs_dir=self.runs_dir)
        logger.on_exp_start(exp)

        # execute
        for epoch in range(nepochs):
            exp.nepochs = epoch
            exp.train_loss_hist.append(tloss[epoch])
            exp.val_loss_hist.append((epoch, vloss[epoch]))
            logger.on_epoch_end(exp=exp, train_loss_epoch=tloss[epoch])
        
        return exp

    def test_writes_one_checkpoint(self):
        nepochs = 1
        exp = self.setup(nepochs)

        # validate
        cps_dir = self.checkpoints_dir(exp)
        md_path = Path(cps_dir, "metadata.json")
        self.assertTrue(md_path.exists())

        cp_paths = [path for path in Path(cps_dir).iterdir() if path.name.endswith(".ckpt")]
        print("cp_paths:")
        print("  " + "\n  ".join(map(str, cp_paths)))
        self.assertEqual(1, len(cp_paths))
        cp_path = cp_paths[0]
        self.assertTrue(cp_path.name.startswith("epoch_0000"))

    def test_writes_checkpoints_2epochs(self):
        nepochs = 2
        exp = self.setup(nepochs)

        # validate
        cps_dir = self.checkpoints_dir(exp)
        md_path = Path(cps_dir, "metadata.json")
        self.assertTrue(md_path.exists())

        cp_paths = sorted([path for path in Path(cps_dir).iterdir() if path.name.endswith(".ckpt")])
        # print("cp_paths:")
        # print("  " + "\n  ".join(map(str, cp_paths)))
        self.assertEqual(2, len(cp_paths))

        path0, path1 = cp_paths
        self.assertTrue(path0.name.startswith("epoch_0000"))
        self.assertTrue(path1.name.startswith("epoch_0001"))

    def test_writes_checkpoints_10epochs(self):
        # setup & execute
        nepochs = 10
        exp = self.setup(nepochs)

        # validate
        cps_dir = self.checkpoints_dir(exp)
        md_path = Path(cps_dir, "metadata.json")
        self.assertTrue(md_path.exists())

        # tloss = cat([up, down]).. so best loss will be at the beginning
        # vloss = cat([down, up]).. so best loss will be in the middle

        # check filesystem.
        cp_paths = sorted([path for path in Path(cps_dir).iterdir() if path.name.endswith(".ckpt")])
        self.assertEqual(2, len(cp_paths))

        path0, path1 = cp_paths
        self.assertTrue(path0.name.startswith("epoch_0000"))
        self.assertTrue(path1.name.startswith("epoch_0004"))

        # check metadata
        loaded = cputil.load_from_metadata(md_path=md_path)
        self.assertEqual(2, len(loaded.runs))
        
        first, second = loaded.runs
        self.assertEqual(0, first.checkpoint_nepochs)
        self.assertEqual(4, second.checkpoint_nepochs)
        
        self.assertTrue(str(first.checkpoint_path.name).startswith("epoch_0000"))
        self.assertTrue(str(second.checkpoint_path.name).startswith("epoch_0004"))
