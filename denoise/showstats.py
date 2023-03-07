# %%
import sys
import datetime

sys.path.append("..")
from experiment import Experiment


import torchsummary
import torch
import model

if __name__ == "__main__":
    # filename = sys.argv[1]
    filename = "runs/denoise_conv_encdec2_1000-20230307-010111/checkpoints/conv_encdec2_k3-s2-op1-p1-c8,c16,c32,c64,emblen_256,nlin_4,hidlen_256,slr_1.0E-03,batch_64,cnt_1,nparams_17.302M,epoch_0877,vloss_0.14950.ckpt"
    with open(filename, "rb") as file:
        state_dict = torch.load(file)
        exp = Experiment.new_from_state_dict(state_dict)
        net = model.ConvEncDec.new_from_state_dict(state_dict["net"]).to("cuda")

        print(net)
        print("---")
        # net = model.ConvEncDec(128, 1, 1, 1, False, True, [ConvDesc], 3, "cuda")
        # net.load_state_dict(state_dict)

        torchsummary.summary(net, input_size=(3, 128, 128), batch_size=1)
        print(exp)
        for field, value in state_dict.items():
            if type(value) in [int, float, bool, datetime.datetime]:
                print(f"{field} = {value}")

# %%
