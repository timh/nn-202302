from typing import List, Dict, Callable
import itertools

from torch import nn

from nnexp.experiment import Experiment
from nnexp.denoise.models import vae, unet

# these are assumed to be defined when this config is eval'ed.
exps: List[Experiment]
vae_net: vae.VarEncDec

def lazy_net_unet(kwargs: Dict[str, any]) -> Callable[[Experiment], nn.Module]:
    def fn(_exp: Experiment) -> nn.Module:
        return unet.Unet(**kwargs)
    return fn
    
twiddles = itertools.product(
    # [False, True],             # self_condition
    [False],             # self_condition
    [                          # dim_mults
        # [1, 2, 4],
        # [1, 2, 4, 8],
        # [1, 2, 4, 8, 16]
        [4, 8, 16],
        # [8, 16, 32],
        # [4, 8, 16, 32],
        # [2, 4, 8, 16]
    ],
    # [1, 4],                     # resnet_block_groups
    # [4, 8],                     # resnet_block_groups
    [4],                         # resnet_block_groups
    # [8],                         # resnet_block_groups
    # ["l2", "l1", "l1_smooth"]        # loss_type
    # ["l2", "l1_smooth"]        # loss_type
    # ["l1", "l1_smooth"],        # loss_type
    ["l2"], # "l1_smooth"],
    # ["l1"],        # loss_type
)

for self_condition, dim_mults, resnet_block_groups, loss_type in twiddles:
    exp = Experiment()
    latent_dim = vae_net.latent_dim.copy()
    lat_chan, lat_size, _ = latent_dim

    args = dict(dim=lat_size, 
                dim_mults=dim_mults, 
                self_condition=self_condition, 
                resnet_block_groups=resnet_block_groups, 
                channels=lat_chan)
    print(f"ARGS: {args}")

    label_parts = ["denoise_unet"]
    label_parts.append(f"dim_mults_{'_'.join(map(str, dim_mults))}")
    label_parts.append(f"selfcond_{self_condition}")
    label_parts.append(f"resblk_{resnet_block_groups}")

    exp.label = ",".join(label_parts)
    exp.loss_type = loss_type
    exp.lazy_net_fn = lazy_net_unet(args)

    exps.append(exp)
