
seqlen_values = [128]
wordlen_values = [2]
nhead_values = [4]
nlayers_values = [4]
emblen_values = [384]
scheduler_values = ["StepLR"]
dropout = 0.2

batch_mini_epochs_values = [
    (128, 2, cfg.nepochs),
]

lrparams_values = [
    ("adamw", 1e-3, 1e-4),
]

all_exp = [
    TextExperiment(seqlen=seqlen, wordlen=wordlen, nhead=nhead, nlayers=nlayers,
                   emblen=emblen, hidlen=emblen * 4,
                   optim_type=lrparams[0], sched_type=sched, startlr=lrparams[1], endlr=lrparams[2], 
                   batch=bme[0], minicnt=bme[1], epochs=bme[2],
                   dropout=dropout)

    # most quickly changing should be at top:
    for lrparams in lrparams_values
    for sched in scheduler_values
    for emblen in emblen_values
    for nlayers in nlayers_values
    for nhead in nhead_values
    for wordlen in wordlen_values
    for seqlen in seqlen_values
    for bme in batch_mini_epochs_values
]
random.shuffle(all_exp)
