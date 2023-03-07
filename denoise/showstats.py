# %%
import sys
import torchsummary
import torch
import model

if __name__ == "__main__":
    # filename = sys.argv[1]
    filename = "runs/denoise_conv_encdec2_0500-20230307-000242/checkpoints/conv_encdec2_k3-s2-op1-p1-c32,c64,c64,emblen_512,nlin_0,hidlen_128,bnorm,slr_1.0E-03,elr_1.0E-03,batch_64,cnt_1,nparams_17.302M,epoch_0499,vloss_0.17592.ckpt"
    with open(filename, "rb") as file:
        state_dict = torch.load(file)
        net = model.ConvEncDec.new_from_state_dict(state_dict).to("cuda")
        print(net)
        print("---")
        # net = model.ConvEncDec(128, 1, 1, 1, False, True, [ConvDesc], 3, "cuda")
        # net.load_state_dict(state_dict)

        torchsummary.summary(net, input_size=(3, 128, 128), batch_size=1)

# %%
