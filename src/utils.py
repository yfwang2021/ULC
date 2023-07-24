import numpy as np
import torch
import re


def get_optimizer(p, name, params):
    if name == "SGD":
        if "momentum" not in params:
            params["momentum"] = 0.0
        if "nesterov" not in params:
            params["nesterov"] = False
        if params["method"] == "MetaMLP":
            return torch.optim.SGD(p, lr=params["lr"],
                   momentum=params["momentum"],
                   nesterov=params["nesterov"],
                   weight_decay=0
                   )
        else:
            return torch.optim.SGD(p, lr=params["lr"],
                   momentum=params["momentum"],
                   nesterov=params["nesterov"],
                   weight_decay=params["l2_reg"]
                   )
    elif name == "Adam":
        if params["method"] == "MetaMLP":
            return torch.optim.Adam(p, lr=params["lr"],
                        weight_decay=0
                        )
        else:
            return torch.optim.Adam(p, lr=params["lr"],
                        weight_decay=params["l2_reg"]
                        )
    elif name == "LBFGS":
        return torch.optim.LBFGS(p, lr=params["lr"], max_iter=1)
    elif name == "AdamW":
        if params["method"] == "MetaMLP":
            return torch.optim.AdamW(p, lr=params["lr"],
                        weight_decay=0
                        )
        else:
            return torch.optim.AdamW(p, lr=params["lr"],
                        weight_decay=params["l2_reg"]
                        )


def parse_float_arg(input, prefix):
    p = re.compile(prefix+"_[+-]?([0-9]*[.])?[0-9]+")
    m = p.search(input)
    if m is None:
        return None
    input = m.group()
    p = re.compile("[+-]?([0-9]*[.])?[0-9]+")
    m = p.search(input)
    return float(m.group())
