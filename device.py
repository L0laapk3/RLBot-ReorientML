import torch
import __main__


if __main__.__file__ == "train_aerial_turn.py":
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")