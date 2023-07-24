import torch


def stable_log1pex(x):
    return -torch.where(x < 0, x, torch.zeros_like(x)) + torch.log(1+torch.exp(-torch.abs(x)))

def cross_entropy_loss(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    x = torch.reshape(x, (-1,))
    z = z.float()
    loss = torch.mean(
        torch.nn.functional.binary_cross_entropy_with_logits(x, z))
    return {"loss": loss}

def exp_delay_loss(targets, outputs, params=None):
    z = torch.reshape((targets["label"][:, 0].float()), (-1, 1))
    x = outputs["logits"]
    lamb = torch.nn.functional.softplus(outputs["log_lamb"])
    log_lamb = torch.log(lamb)
    d = torch.reshape(targets["label"][:, 1].float(), (-1, 1))
    e = d
    p = torch.nn.Sigmoid()(x)
    pos_loss = -(-stable_log1pex(x) + log_lamb - lamb*d)
    neg_loss = -torch.log(1 - p + p*torch.exp(-lamb*e))
    return {"loss": torch.mean(pos_loss*z + neg_loss*(1-z))}

def fsiw_loss(targets, outputs, params=None):
    x = outputs["logits"]
    logits0 = outputs["logits0"].float().detach()
    logits1 = outputs["logits1"].float().detach()
    prob0 = torch.nn.Sigmoid()(logits0)
    prob1 = torch.nn.Sigmoid()(logits1)
    z = torch.reshape(targets["label"].float(), (-1, 1))

    pos_loss = stable_log1pex(x)
    neg_loss = x + stable_log1pex(x)

    pos_weight = 1/(prob1+1e-8)
    neg_weight = prob0

    clf_loss = torch.mean(
        pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))
    loss = clf_loss
    return {
        "loss": loss,
    }

def get_loss_fn(name):
    if name == "cross_entropy_loss":
        return cross_entropy_loss
    elif name == "delayed_feedback_loss":
        return exp_delay_loss
    elif name == "fsiw_loss":
        return fsiw_loss
    elif name == "non_negative_loss":
        return None
    else:
        raise NotImplementedError("{} loss does not implemented".format(name))
