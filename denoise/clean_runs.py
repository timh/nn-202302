import sys
import re
from typing import Tuple, List, Set, Dict
from collections import OrderedDict
from pathlib import Path
import argparse
import datetime

sys.path.append("..")
import experiment
from experiment import Experiment
import model_util
import checkpoint_util
import cmdline

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
    
    # def list_checkpoints(self) -> List[Tuple[Path, Experiment]]:
    #     return checkpoint_util.find_checkpoints()

def sort_key(path: Path):
    cnt = str(path).count("/")
    return (-cnt, str(path))

def _filter_results(paths: List[Path], cfg: Config) -> List[Path]:
    if cfg.pattern:
        paths = [path for path in paths
                 if cfg.pattern.match(str(path))]
    
    if cfg.ignore_pattern:
        paths = [path for path in paths
                 if not cfg.ignore_pattern.match(str(path))]

    return paths        

class State:
    remove: Dict[Path, List[Path]]
    keep: Dict[Path, List[Path]]
    contents: Dict[Path, List[Path]]
    move: Dict[Path, Path]

    def __init__(self, cfg: Config):
        self.contents = dict()
        self.remove = dict()
        self.keep = dict()
        self.move = dict()

        def walk(pdir: Path, root = False):
            contents = [path for path in pdir.iterdir()]
            if root:
                contents = [path for path in contents
                            if path.name != 'all_images' and
                            not path.name.startswith("backup")]
                contents = [path for path in contents
                            if path.is_dir()]

            contents = _filter_results(contents, cfg)
            contents = sorted(contents, key=lambda path: path.lstat().st_ctime)
            file_contents = [path for path in contents if not path.is_dir()]

            self.remove[pdir] = list(file_contents)
            self.keep[pdir] = list()
            self.contents[pdir] = list(contents)

            for sub in contents:
                if sub.is_dir():
                    walk(sub)
        
        walk(Path("runs"), root=True)
    
    def get_remove_files(self, cfg: Config) -> List[Path]:
        all_files = [path
                     for pdir, pdir_remove in self.remove.items()
                     for path in pdir_remove]
        all_files = _filter_results(all_files, cfg)
        return sorted(all_files, key=sort_key)
    
    def get_remove_dirs(self, cfg: Config) -> List[Path]:
        all_dirs = [pdir
                    for pdir, pdir_keep in self.keep.items()
                    if len(pdir_keep) == 0]
        all_dirs = _filter_results(all_dirs, cfg)
        return sorted(all_dirs, key=sort_key)
    
    def get_moves(self, cfg: Config) -> List[Tuple[Path, Path]]:
        all_move_from = _filter_results(self.move.keys(), cfg)
        all_move_from = sorted(all_move_from, key=sort_key)
        return [(move_from, self.move[move_from]) 
                for move_from in all_move_from]
    
    # returns both files and directories.
    def get_contents(self, pdir: Path) -> List[Path]:
        if pdir not in self.contents:
            return list()

        return list(self.contents[pdir])

    def mark(self, path: Path):
        if path.is_dir():
            raise ValueError(f"don't call this on directories: {path}")
        
        pdir = path.parent
        if path in self.remove[pdir]:
            self.remove[pdir].remove(path)
            self.keep[pdir].append(path)

            while pdir != Path("runs"):
                self.keep[pdir.parent].append(pdir)
                pdir = pdir.parent
    
    def mark_move(self, path: Path, to_path: Path):
        pdir = path.parent
        self.move[path] = to_path
        self.mark(path)
    
    def markdir(self, path: Path):
        if not path.is_dir():
            raise ValueError(f"don't call this on files: {path}")
        
        for content in self.get_contents(path):
            self.mark(content)

def mark_checkpoint(cfg: Config, state: State, cp_path: Path):
    pdir = cp_path.parent
    if pdir.name != 'checkpoints':
        raise ValueError(f"should only be called within checkpoints, but it's {cp_path}")

    name = cp_path.name
    idx = name.index(",epoch")
    basename = name[:idx]

    state.mark(cp_path)

    # mark related JSON & latent checkpoints.
    for other in state.get_contents(pdir):
        if other.name.startswith(basename):
            state.mark(other)
    
    # mark related tensorboard files.
    rundir = cp_path.parent.parent
    for path in state.get_contents(rundir):
        if path.is_dir() and path.name.endswith(basename):
            # mark all of these tensorboard contents to be saved.
            state.markdir(path)
            continue

        if path.is_dir():
            continue

        if path.name.endswith(".status"):
            if path.name.startswith(basename):
                # status file for the checkpoint
                state.mark(path)

        elif path.name == 'run-progress.png':
            to_path = Path("runs", "backup-run-progress", path.parent.name + "-run-progress.png")
            state.mark_move(path, to_path)
        
        elif path.name.startswith("events"):
            # tensorboard cruft
            pass

        else:
            print(f"what do I do? {path}")

    # mark related images.
    imagedir = Path(rundir, 'images')
    for path in state.get_contents(imagedir):
        if path.name.startswith(basename):
            state.mark(path)
        # else don't.
    
    
def clean(cfg: Config, state: State):
    checkpoints = cfg.list_checkpoints(dedup_runs=False)
    cp_exps = [exp for _path, exp in checkpoints]
    cp_paths = [path for path, _exp in checkpoints]

    cps_by_root = checkpoint_util.find_resume_roots(checkpoints)

    # mark only the root checkpoints.
    for root, offspring_idxs in cps_by_root.items():
        mark_checkpoint(cfg, state, cp_paths[root])
        # if len(offspring_idxs) == 1:
        #     continue
        # for offspring in offspring_idxs:
        #     mark_checkpoint(cfg, state, cp_paths[offspring])

def do_remove_files(cfg: Config, state: State):
    if cfg.just_print_dirs:
        return

    files = state.get_remove_files(cfg)
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

def do_move_files(cfg: Config, state: State):
    if cfg.just_print_dirs:
        return

    to_move = state.get_moves(cfg)
    if len(to_move):
        dry_run = " (dry run)"
        if cfg.doit:
            dry_run = ""

        print(f"move {len(to_move)} files:{dry_run}")
        for move_from, move_to in to_move:
            print(f"  {move_from} -> {move_to}")
            if cfg.doit:
                move_to.parent.mkdir(exist_ok=True, parents=True)
                move_from.rename(move_to)

def do_remove_dirs(cfg: Config, state: State):
    if cfg.just_print_files:
        return
    
    dirs = state.get_remove_dirs(cfg)
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

    state = State(cfg)
    clean(cfg, state)

    do_remove_files(cfg, state)
    do_move_files(cfg, state)
    do_remove_dirs(cfg, state)
