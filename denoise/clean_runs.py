import sys
import re
from typing import Tuple, List, Set, Deque, Dict, DefaultDict
from collections import defaultdict, deque, OrderedDict
from pathlib import Path
import argparse
import datetime

sys.path.append("..")
import experiment
from experiment import Experiment
import model_util
import cmdline

EXTRA_IGNORE_FIELDS = \
    set('max_epochs batch_size label sched_args optim_args '
        'do_compile use_amp'.split())

IGNORE_FIELDS = experiment.SAME_IGNORE_FIELDS | EXTRA_IGNORE_FIELDS

class Config(cmdline.QueryConfig):
    start_time: datetime.datetime
    doit: bool
    verbose: bool

    ignore_pattern: re.Pattern
    ignore_newer_than: datetime.datetime

    just_print_dirs: bool
    just_print_files: bool

    def __init__(self):
        super().__init__()
        self.add_argument("--doit", default=False, action='store_true')
        self.add_argument("--ignore_pattern", type=str, default=None)
        self.add_argument("-n", "--ignore_newer_than", type=int, default=600, help="ignore checkpoints newer than N seconds")
        self.add_argument("-v", "--verbose", default=False, action='store_true')
        self.add_argument("--just_print_dirs", default=False, action='store_true')
        self.add_argument("--just_print_files", default=False, action='store_true')

        self.start_time = datetime.datetime.now()
    
    def parse_args(self) -> 'Config':
        super().parse_args()

        if self.ignore_pattern:
            self.ignore_pattern = re.compile(self.ignore_pattern)
        self.ignore_newer_than = self.start_time - datetime.timedelta(seconds=self.ignore_newer_than)

        if self.verbose:
            self.doit = False

        return self

class State:
    orig_contents: Dict[Path, Set[Path]]
    contents: Dict[Path, Set[Path]]
    remove_files: Dict[Path, Set[Path]]
    remove_dirs: Set[Path]
    verbose: bool

    def __init__(self, verbose: bool = False):
        self.contents = dict()
        self.orig_contents = dict()
        self.remove_files = OrderedDict()
        self.remove_dirs = set()
        self.verbose = verbose

        def walk(pdir: Path, is_runs_dir = False):
            contents = set(pdir.iterdir())
            if is_runs_dir:
                contents = [path for path in contents 
                            if path.is_dir() and path.name != "all_images"]
                contents = sorted(contents, key=lambda path: path.stat().st_ctime)

            self.contents[pdir] = set(contents)
            self.orig_contents[pdir] = set(contents)
            self.remove_files[pdir] = set()

            if len(contents) == 0 and not is_runs_dir:
                self.remove_dirs.add(pdir)

            for sub in contents:
                if not sub.is_dir():
                    continue
                walk(sub)
        
        walk(Path("runs"), True)
    
    def get_run_dirs(self) -> Set[Path]:
        return self.get_contents(Path("runs"))

    def get_contents(self, pdir: Path) -> Set[Path]:
        if pdir not in self.contents:
            return set()
        return self.contents[pdir]

    """
    return files scheduled to be removed
    """
    def get_remove_files(self) -> List[Path]:
        res: List[Path] = list()
        for pdir, files in self.remove_files.items():
            res.extend(files)
        return res
    
    def will_remove(self, file: Path) -> bool:
        pdir = file.parent
        if pdir not in self.remove_files:
            return False
        return file in self.remove_files[pdir]

    def append(self, path: Path):
        pdir = path.parent

        if self.contents[pdir] == set([path]):
            self.append(pdir)

        self.contents[pdir].remove(path)

        if path.is_dir():
            self.remove_dirs.add(path)
        else:
            self.remove_files[pdir].add(path)

    
    def extend(self, paths: List[Path]):
        for file in paths:
            self.append(file)

    def get_remove_dirs(self) -> Set[Path]:
        def _show_dir(pdir: Path, level = 0):
            contents = self.get_contents(pdir)
            orig_contents = self.orig_contents[pdir]

            indent = "  " * level
            if level == 0:
                if pdir in self.remove_dirs:
                    print(f"{indent}\033[1;31m{pdir}\033[0m:")
                else:
                    print(f"{indent}\033[1m{pdir}\033[0m:")
            for path in orig_contents:
                is_dir = ":" if path.is_dir() else ""
                if path in contents:
                    print(f"{indent}  \033[1m{path.name}\033[0m{is_dir}")
                else:
                    print(f"{indent}  \033[1;31m{path.name}\033[0m{is_dir}")

                if path.is_dir():
                    _show_dir(path, level + 1)
                
            print()

        if self.verbose:
            for run_dir in self.get_run_dirs():
                _show_dir(run_dir)
        return self.remove_dirs

def _find_latent_checkpoints(state: State, cp_path: Path) -> List[Path]:
    cp_basename = str(cp_path.name).replace(".ckpt", "")

    res: List[Path] = list()
    for path in state.get_contents(cp_path.parent):
        if cp_basename not in str(path) or ".lat-n" not in str(path) or path.is_dir():
            continue
        res.append(path)
    
    return res

def clean_resumed(cfg: Config, state: State):
    # index (key) is the same as (values indexes)
    # key = inner, values = outer
    checkpoints = cfg.list_checkpoints()

    cp_paths = [cp_path for cp_path, _cp_exp in checkpoints]
    cp_exps = [cp_exp for _cp_path, cp_exp in checkpoints]

    # build up mapping of "inner is the same as outer" pairs.
    cps_same: Dict[int, int] = dict()
    for outer_idx, outer_exp in enumerate(cp_exps):
        for inner_idx in range(outer_idx + 1, len(cp_exps)):
            inner_exp = cp_exps[inner_idx]

            if outer_exp.label != inner_exp.label:
                continue

            if inner_idx in cps_same:
                continue

            is_same, _same_fields, diff_fields = \
                outer_exp.is_same(inner_exp, extra_ignore_fields=EXTRA_IGNORE_FIELDS,
                                  return_tuple=True)
            if is_same:
                cps_same[inner_idx] = outer_idx

    # build up a dict of all the lineages of checkpoints:
    # root idx -> all child indexes
    def find_root(idx: int) -> int:
        if idx not in cps_same:
            return idx
        return find_root(cps_same[idx])        

    cps_by_root: DefaultDict[int, List[int]] = defaultdict(list)
    for idx in range(len(cp_exps)):
        root = find_root(idx)
        cps_by_root[root].append(idx)

    # identify files to remove
    for root, offspring_idxs in cps_by_root.items():
        if len(offspring_idxs) == 1:
            continue

        pairs = [(idx, cp_exps[idx].nepochs) for idx in offspring_idxs]
        pairs_by_nepochs = list(reversed(sorted(pairs, key=lambda pair: pair[1])))

        # pairs_slist = [
        #     f"{idx}:{nepochs}"
        #     for idx, nepochs in pairs_by_nepochs
        # ]
        # pairs_s = " ".join(pairs_slist)
        # print(f"root {root:3}: {pairs_s}")

        keep_idxs = pairs_by_nepochs[0][0]
        remove_idxs = [pair[0] for pair in pairs_by_nepochs[1:]]
        # print(f"  remove: " + " ".join(map(str, remove_idxs)))

        for one_remove_idx in remove_idxs:
            cp_path = cp_paths[one_remove_idx]
            json_path = Path(str(cp_path).replace(".ckpt", ".json"))

            to_remove = [cp_path, json_path]
            to_remove.extend(_find_latent_checkpoints(state, cp_path))

            cp_name_str = str(cp_path.name)
            if ",epoch_" in cp_name_str:
                cp_minus_epoch = cp_name_str[:cp_name_str.index(",epoch_")]

                run_dir = Path(cp_path.parent.parent)

                status_path = Path(run_dir, cp_minus_epoch + ".status")
                if status_path in state.get_contents(status_path.parent):
                    to_remove.append(status_path)
                
                progress_path = Path(run_dir, "images", cp_minus_epoch + "-progress.png")
                if progress_path in state.get_contents(progress_path.parent):
                    to_remove.append(progress_path)

            state.extend(to_remove)

def _tensorboard_paths(cfg: Config, runpath: Path) -> List[Path]:
    res: List[Path] = list()
    for path in runpath.iterdir():
        if not path.is_dir() and path.name.startswith("events"):
            res.append(path)
            continue

        if not path.is_dir() or path.name in ["checkpoints", "images"]:
            continue

        dir_files = [file for file in path.iterdir() if file.name.startswith("events")]
        dir_files = [file for file in dir_files 
                     if datetime.datetime.fromtimestamp(file.stat().st_mtime) < cfg.ignore_newer_than]
        if not dir_files:
            continue

        res.extend(dir_files)
    
    if cfg.pattern is not None:
        res = [file for file in res if cfg.pattern.match(str(file))]
    if cfg.ignore_pattern is not None:
        res = [file for file in res if not cfg.ignore_pattern.match(str(file))]

    return res


# files within a 'runs' subdir, all related to the same experiment:
# tensorboard logs:    batch_lr_k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M
#        done file:             k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M.status
#       checkpoint: checkpoints/k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M,epoch_0047.ckpt
#         metadata: checkpoints/k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M,epoch_0047.json
#   progress image: images/k3-s1-c64,c64,mp2-c128,mp2-c256,mp2-c512,mp2-c1024,emblen_0,nlin_0,hidlen_384,bias_False,loss_edge+l1,batch_128,slr_1.0E-03,elr_1.0E-04,nparams_22.310M-progress.png
RE_TENSORBOARD = re.compile(r"([a-z_]+)_(.+)")
def clean_tensorboard(cfg: Config, state: State):
    now = datetime.datetime.now()

    checkpoints = cfg.list_checkpoints()

    # look for tensorboard files to remove
    for run_dir in state.get_run_dirs():
        for tb_file in _tensorboard_paths(cfg, run_dir):
            match = RE_TENSORBOARD.match(tb_file.parent.name)
            if match:
                _tb_key, exp_label = match.groups()
            else:
                exp_label = None

            found_checkpoint = False
            for cp_path, cp_exp in checkpoints:
                cp_run_path = cp_path.parent.parent
                if (cp_run_path == run_dir and 
                    cp_exp.label == exp_label and 
                    not state.will_remove(cp_path)):
                    found_checkpoint = True
                    break

            if not found_checkpoint:
                state.append(tb_file)

# run after the others - remove images if they are the last thing in there.
def clean_image_progress(cfg: Config, state: State):
    for rundir in list(state.get_run_dirs()):
        contents = state.get_contents(rundir)
        if len(contents) != 1:
            continue
        first = list(contents)[0]
        if first.name != 'run-progress.png':
            continue
        state.append(first)

def do_remove_files(cfg: Config, state: State):
    if cfg.just_print_dirs:
        return

    files = state.get_remove_files()
    if cfg.just_print_files:
        print("\n".join(map(str, files)))
        return

    if len(files):
        dry_run = " (dry run)"
        if cfg.doit:
            dry_run = ""

        print(f"remove {len(files)} files:{dry_run}")
        for rmfile in files:
            print(f"  {rmfile}")
            if cfg.doit:
                rmfile.unlink()

def do_remove_dirs(cfg: Config, state: State):
    if cfg.just_print_files:
        return
    
    dirs = state.get_remove_dirs()
    if cfg.just_print_dirs:
        print("\n".join(map(str, dirs)))
        return
    
    if len(dirs):
        dry_run = " (dry run)"
        if cfg.doit:
            dry_run = ""

        print(f"remove {len(dirs)} directories:{dry_run}")
        for rmdir in dirs:
            print(f"  {rmdir}")
            if cfg.doit:
                rmdir.rmdir()
    
if __name__ == "__main__":
    cfg = Config().parse_args()

    state = State(cfg.verbose)
    clean_resumed(cfg, state)
    clean_tensorboard(cfg, state)
    clean_image_progress(cfg, state)

    do_remove_files(cfg, state)
    do_remove_dirs(cfg, state)

    # tb_files, tb_dirs = clean_tensorboard(cfg)
    # do_remove()
