import sys
import re
from typing import Tuple, List, Set
from pathlib import Path
import argparse
import datetime

sys.path.append("..")
import experiment
from experiment import Experiment
import denoise_logger

# files within a 'runs' subdir, all related to the same experiment:
# tensorboard logs:    batch_lr_k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M
#        done file:             k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M.status
#       checkpoint: checkpoints/k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M,epoch_0047.ckpt
#         metadata: checkpoints/k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M,epoch_0047.json
#   progress image: images/k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M-progress.png
RE_TENSORBOARD = re.compile(r"([\w_]+\w+)_(.*)")
def clean_unfinished(checkpoints: List[Tuple[Path, Experiment]], cfg: argparse.Namespace):
    now = datetime.datetime.now()
    threshold = datetime.timedelta(seconds=cfg.threshold)

    # pathandexp_by_label = {exp.label: (cp_path, exp) for cp_path, exp in checkpoints}
    exp_finished = {exp.label for cp_path, exp in checkpoints 
                    if Path(str(cp_path.parent.parent), exp.label + ".status").exists()}
    exp_toonew = {exp.label: (now - exp.curtime) for cp_path, exp in checkpoints
                  if (now - exp.curtime) <= threshold}

    def tensorboard_paths(runpath: Path) -> List[Path]:
        res: List[Path] = list()
        for tb_dir in runpath.iterdir():
            match = RE_TENSORBOARD.match(tb_dir.name)
            if not tb_dir.is_dir() or not match:
                continue

            tb_prefix, exp_label = match.groups()
            if exp_label in exp_finished:
                if cfg.verbose:
                    print(f"ignore finished:\n  {exp_label}\n  {str(tb_dir)}")
                    print()
                continue
            if exp_label in exp_toonew:
                if cfg.verbose:
                    print(f"ignore toonew:\n  {exp_label}\n  updated {exp_toonew[exp_label]=} ago\n  {str(tb_dir)}")
                    print()
                continue
            
            res.append(tb_dir)
        return res

    # rundir, files_to_remove, dirs_to_remove
    to_remove: List[Tuple[Path, List[Path], List[Path]]] = list()
    for runpath in Path("runs").iterdir():
        if runpath.is_dir():
            tb_paths = tensorboard_paths(runpath)
            if not tb_paths:
                continue

            remove_files: List[Path] = list()
            remove_dirs: List[Path] = list()
            for tb_path in tb_paths:
                for content in tb_path.iterdir():
                    if not content.name.startswith("events.") or not content.is_file():
                        raise Exception(f"logic error: \n  {content=}")
                    remove_files.append(content)
                remove_dirs.append(tb_path)
            
            entry = (runpath, remove_files, remove_dirs)
            to_remove.append(entry)
    
    for runpath, remove_files, remove_dirs in to_remove:
        dry_run = " (dry run)" if not cfg.doit else ""
        print(f"{runpath}:")
        for fpath in remove_files:
            print(f"  \033[1mremove file {fpath}\033[0m{dry_run}")
            if cfg.doit:
                fpath.unlink()
        for dpath in remove_dirs:
            print(f"  \033[1mremove  dir {dpath}\033[0m{dry_run}")
            if cfg.doit:
                dpath.rmdir()
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doit", default=False, action='store_true')
    parser.add_argument("-p", "--pattern", type=str, default=None)
    parser.add_argument("--threshold", type=int, default=60, help="threshold (in seconds): any exp newer than this will be ignored")
    parser.add_argument("-v", "--verbose", default=False, action='store_true')

    cfg = parser.parse_args()
    checkpoints = denoise_logger.find_all_checkpoints(Path("runs"))
    if cfg.pattern:
        checkpoints = [cp for cp in checkpoints if cfg.pattern in str(cp[0])]

    clean_unfinished(checkpoints, cfg)