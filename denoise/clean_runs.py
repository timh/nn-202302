import sys
import re
from typing import Tuple, List, Set
from pathlib import Path
import argparse
import datetime

sys.path.append("..")
import experiment
from experiment import Experiment
import model_util

class Config:
    include_pattern: re.Pattern
    ignore_pattern: re.Pattern
    ignore_newer_than: datetime.datetime
    doit: bool

def tensorboard_paths(cfg: Config, runpath: Path) -> List[Tuple[Path, List[Path]]]:
    res: List[Tuple[Path, List[Path]]] = list()
    for tb_dir in runpath.iterdir():
        if not tb_dir.is_dir() or tb_dir.name in ["checkpoints", "images"]:
            continue

        dir_files = [file for file in tb_dir.iterdir() if file.name.startswith("events")]
        if cfg.include_pattern is not None:
            dir_files = [file for file in dir_files if cfg.include_pattern.match(str(file))]
        if cfg.ignore_pattern is not None:
            dir_files = [file for file in dir_files if not cfg.ignore_pattern.match(str(file))]

        dir_files = [file for file in dir_files 
                     if datetime.datetime.fromtimestamp(file.stat().st_mtime) < cfg.ignore_newer_than]

        if not dir_files:
            continue

        res.append((tb_dir, dir_files))

    return res


# files within a 'runs' subdir, all related to the same experiment:
# tensorboard logs:    batch_lr_k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M
#        done file:             k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M.status
#       checkpoint: checkpoints/k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M,epoch_0047.ckpt
#         metadata: checkpoints/k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M,epoch_0047.json
#   progress image: images/k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M-progress.png
RE_TENSORBOARD = re.compile(r"([a-z_]+)_(.+)")
def clean_unfinished(cfg: argparse.Namespace, checkpoints: List[Tuple[Path, Experiment]]):
    now = datetime.datetime.now()

    # rundir, files_to_remove, dirs_to_remove
    to_remove_dirs: List[Path] = list()
    to_remove_files: List[Path] = list()
    run_dirs = sorted(list(Path("runs").iterdir()), key=lambda path: path.stat().st_ctime)
    for runpath in run_dirs:
        if not runpath.is_dir():
            continue

        tb_paths = tensorboard_paths(cfg, runpath)

        for tb_dir, tb_files in tb_paths:
            found_checkpoint = False
            match = RE_TENSORBOARD.match(tb_dir.name)
            if not match:
                continue
            tb_key, exp_label = match.groups()
            for cp_path, cp_exp in checkpoints:
                if cp_exp.label.startswith(exp_label):
                    found_checkpoint = True
                    break
            if not found_checkpoint:
                to_remove_dirs.append(tb_dir)
                to_remove_files.extend(tb_files)
    
    if cfg.doit:
        dry_run = ""
    else:
        dry_run = " (dry run)"
    if len(to_remove_files):
        print(f"remove {len(to_remove_files)} files:{dry_run}")
        for rmfile in to_remove_files:
            print(f"  {rmfile}")
            if cfg.doit:
                rmfile.unlink()
    
    if len(to_remove_dirs):
        print(f"remove {len(to_remove_dirs)} directories:{dry_run}")
        for rmdir in to_remove_dirs:
            print(f"  {rmdir}")
            if cfg.doit:
                rmdir.rmdir()
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doit", default=False, action='store_true')
    parser.add_argument("-p", "--include_pattern", type=str, default=None)
    parser.add_argument("--ignore_pattern", type=str, default=None)
    parser.add_argument("-n", "--ignore_newer_than", type=int, default=600, help="ignore checkpoints newer than N seconds")
    parser.add_argument("-v", "--verbose", default=False, action='store_true')

    cfg: Config = parser.parse_args(namespace=Config())
    if cfg.include_pattern:
        cfg.include_pattern = re.compile(cfg.include_pattern)
    if cfg.ignore_pattern:
        cfg.ignore_pattern = re.compile(cfg.ignore_pattern)
    cfg.ignore_newer_than = datetime.datetime.now() - datetime.timedelta(seconds=cfg.ignore_newer_than)

    checkpoints = model_util.find_checkpoints(only_paths=cfg.include_pattern)
    # checkpoints = reversed(sorted(checkpoints, key=lambda cp_pair: cp_pair[1].saved_at))

    clean_unfinished(cfg, checkpoints)