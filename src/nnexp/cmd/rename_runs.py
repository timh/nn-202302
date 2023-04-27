from pathlib import Path
from typing import Dict, List, Tuple, Set
import re
import json
import os

from nnexp import checkpoint_util
from nnexp.experiment import Experiment

# checkpoint & tensorboard subdir are the same format:
# 20230329-132815-bqzdnw--type_unet,img_latdim_8_64_64,noisefn_normal,noise_cosine_300,dim_mults_1_2_4,selfcond_False,resblk_4
RE_BASE = re.compile(r"([0-9\-]+-)(\w+)(--.*)")

# images:
# run-progress--abthsc,bjkaqp,bqzdnw,bzwfuz,cestcm,cpdoxh,fhilug,iypgbt,jlrtzp,ksblyl,ladkab,lchjco,rmtfau,sbhdjw,soojec,tcqqje,tvrbtp,tymwdb,tyshvi,usruho,vgjwvq,vndnhl,wnmrto--20230330-050310.png
RE_IMAGE = re.compile(r"(run-progress--)([\w,]+)(--.*)")

# anim_20230329-141803-soojec,nepochs_2,stepsper_150.mp4
RE_ANIM = re.compile(r"(anim_[0-9\-]+-)(\w+)(,.*)")

# ROOT_BACKUP = Path("runs.0331")
ROOT_RUNS = checkpoint_util.DEFAULT_DIR
ROOT_BACKUP = Path(ROOT_RUNS.parent, "runs.0427")

def backup_to_runs(backup_path: Path) -> Path:
    new_parts: List[str] = list()
    for part in backup_path.parts:
        if part == ROOT_BACKUP.name:
            part = ROOT_RUNS.name
        new_parts.append(part)
    return Path(*new_parts)

def copy_paths(backup_dir: Path, pat: re.Pattern, 
               shortcode_remap: Dict[str, str], shortcode_same: Set[str], 
               copies: List[Tuple[Path, Path]]):
    new_dir = backup_to_runs(backup_dir)
    for backup_path in backup_dir.iterdir():
        match = pat.match(backup_path.name)
        if not match:
            # print(f"copy_paths: {backup_path} doesn't match {pat}")
            continue

        prefix, shortcodes, rest = match.groups()

        old_shortcodes = shortcodes.split(",")
        new_shortcodes: List[str] = list()
        for oldcode in old_shortcodes:
            if oldcode in shortcode_same:
                continue
            newcode = shortcode_remap.get(oldcode, None)
            if not newcode:
                # print(f"dunno shortcode {oldcode}: {backup_path}")
                continue
            new_shortcodes.append(newcode)
        
        if not len(new_shortcodes):
            continue

        new_shortcodes = sorted(new_shortcodes)

        new_filename = prefix + ",".join(new_shortcodes) + rest
        new_path = Path(new_dir, new_filename)
        copies.append((backup_path, new_path))

if __name__ == "__main__":
    copies: List[Tuple[Path, Path]] = list()
    shortcode_remap: Dict[str, str] = dict()
    shortcode_same: Set[str] = set()

    basename = "ae"
    cp_dir = Path(ROOT_BACKUP, f"checkpoints-{basename}")
    image_dir = Path(ROOT_BACKUP, f"images-{basename}")
    tensorboard_dir = Path(ROOT_BACKUP, f"tensorboard-{basename}")
    anim_dir = Path("animations")

    for backup_dir in cp_dir.iterdir():
        if not backup_dir.is_dir():
            continue

        md_path = Path(backup_dir, "metadata.json")
        if not md_path.exists():
            continue

        exp = checkpoint_util.load_from_metadata(md_path)
        match = RE_BASE.match(backup_dir.name)
        if not match and backup_dir.name.startswith("temp"):
            continue

        old_shortcode = match.group(2)
        new_shortcode = exp.shortcode
        shortcode_same.add(new_shortcode)

        new_md_path = backup_to_runs(md_path)

        if old_shortcode == new_shortcode:
            shortcode_same.add(old_shortcode)
            continue

        print(f"{old_shortcode} -> {new_shortcode}")
        shortcode_remap[old_shortcode] = new_shortcode

        rt_exp = Experiment().load_model_dict(exp.metadata_dict())
        rt_shortcode = rt_exp.shortcode
        if rt_exp.shortcode != exp.shortcode:
            diff_fields = ", ".join(exp.id_diff(rt_exp))
            print(f"  diffs: {diff_fields}")
            raise Exception("done")
    
    print("CHECKPOINTS")
    copy_paths(cp_dir, RE_BASE, shortcode_remap=shortcode_remap, shortcode_same=shortcode_same, copies=copies)
    
    print("TENSORBOARD")
    copy_paths(tensorboard_dir, RE_BASE, shortcode_remap=shortcode_remap, shortcode_same=shortcode_same, copies=copies)
    
    print("IMAGES")
    copy_paths(image_dir, RE_IMAGE, shortcode_remap=shortcode_remap, shortcode_same=shortcode_same, copies=copies)

    print("ANIMATIONS")
    renames: List[Tuple[Path, Path]] = list()
    copy_paths(anim_dir, RE_ANIM, shortcode_remap=shortcode_remap, shortcode_same=shortcode_same, copies=renames)

    if len(copies):
        old_paths, new_paths = zip(*copies)

        print("COPIES:")    
        for old_path, new_path in copies:
            if new_path.exists():
                continue
            print(f"{old_path} ->\n{new_path}")
            print()

            if os.environ.get("DOIT") == 'true':
                if old_path.is_dir():
                    new_path.mkdir(exist_ok=True, parents=True)
                    for old_content in old_path.iterdir():
                        if old_content.is_dir():
                            print(f"skip hardlink for dir: {old_content}")
                            continue
                        new_content = Path(new_path, old_content.name)
                        old_content.link_to(new_content)
                    continue

                old_path.link_to(new_path)
                # os.system(f"mv {old_path} {new_path}")
                # old_path.rename(new_path)
                pass

    def new_code(shortcode: str) -> str:
        if shortcode in shortcode_remap:
            return shortcode_remap[shortcode]
        return shortcode
    
    print("CHECKPOINT RESUME:")
    metadata_paths: List[Path] = list()
    for cur_dir in Path(ROOT_RUNS, f"checkpoints-{basename}").iterdir():
        md_path = Path(cur_dir, "metadata.json")
        if md_path.exists():
            metadata_paths.append(md_path)

    for md_path in metadata_paths:
        num_changes = 0
        with open(md_path, "r") as md_file:
            exp_dict = json.load(md_file)
        # exp = checkpoint_util.load_from_json(md_path)
        # print(f"json={exp_dict['shortcode']} {md_path}")

        # exp.shortcode
        old_shortcode = exp_dict['shortcode']
        new_shortcode = new_code(old_shortcode)
        if old_shortcode != new_shortcode:
            num_changes += 1
            exp_dict['shortcode'] = new_shortcode
            print(f"shortcode {old_shortcode} -> {new_shortcode}")
        
        for run_idx, run_dict in enumerate(exp_dict['runs']):
            old_cp_path = run_dict.get('checkpoint_path', None)
            if not old_cp_path:
                continue

            old_cp_path = Path(old_cp_path)
            old_cp_dir = old_cp_path.parent

            match = RE_BASE.match(old_cp_dir.name)
            if not match:
                raise Exception("can't parse {old_cp_dir.name}")

            prefix, old_shortcode, rest = match.groups()
            new_shortcode = new_code(old_shortcode)
            if old_shortcode != new_shortcode:
                num_changes += 1
                new_dirname = prefix + new_shortcode + rest
                new_cp_path = Path(old_cp_dir.parent, new_dirname, old_cp_path.name)
                run_dict['checkpoint_path'] = str(new_cp_path)

                print(f"  run {run_idx + 1}")
                print(f"    {old_cp_dir.name} ->")
                print(f"    {new_dirname}")
                print()
        
        if num_changes > 0 and os.environ.get("DOIT") == 'true':
            with open(md_path, "w") as md_file:
                json.dump(exp_dict, md_file, indent=2)







