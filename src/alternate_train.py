from copy import deepcopy

from models import get_model
from loss import get_loss_fn
from single_train_test import test
from utils import get_optimizer
from metrics import cal_llloss_with_logits
from data import get_criteo_dataset, DelayDataset
from tqdm import tqdm
import numpy as np
import torch
from pretrain import optim_step

def get_valid_llloss_blc(model, test_data, correction_model):
    all_logits = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_data)):
            batch_x = batch[0].to("cuda")
            batch_y = batch[1].to("cuda")
            logits0 = correction_model(batch_x)["logits"]
            corrections = (torch.nn.Sigmoid()(logits0)).flatten()
            corrections = torch.where(batch_y < 1, corrections, batch_y.float())
            logits = model(batch_x)["logits"]
            all_logits.append(logits.cpu().numpy())
            all_labels.append(corrections.cpu().numpy())
    all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1,))
    all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1,))
 

    llloss = cal_llloss_with_logits(all_labels, all_logits)
    return llloss

def seed_everything(params):
    torch.manual_seed(params["seed"])
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(params["seed"])
    np.random.seed(params["seed"])

def alternate_run(params, wandb):
    cvr_model = None
    sub_model = None
    dataset = get_criteo_dataset(params)
    sub_params = deepcopy(params)

    sub_params["dataset"] = "fsiwsg_cd_"+str(params["CD"])+"_end_"+str(params["training_end_day"])+"_seed_"+str(params["seed"])
    np.random.seed(params["seed"])
    sub_dataset = get_criteo_dataset(sub_params)["train"]
    np.random.seed(params["seed"])
        
    params["log_step"] = 0
    params["idx"] = 1
    for i in range(2):
        seed_everything(params)
        sub_model = sub_train(cvr_model, sub_dataset, params)
        seed_everything(params)
        cvr_model = cvr_train(sub_model, dataset, params, wandb)
        params["idx"] += 1

def sub_train(cvr_model, sub_dataset, params):
    train_data_x = torch.from_numpy(sub_dataset["x"].to_numpy().astype(np.float32)).to("cuda")
    train_data_label = torch.from_numpy(sub_dataset["labels"]).to("cuda")
    train_data_label = 1 - train_data_label # the lc label is the reverse of fsiw0
    train_data = torch.utils.data.TensorDataset(train_data_x, train_data_label)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)

    model = get_model("MLP_FSIW", params).to("cuda")

    if cvr_model != None:
        sd = cvr_model.state_dict()
        part_sd = {k:v for k,v in sd.items() if ("category_embeddings" in k) or ("numeric_embeddings" in k)}
        model_dict = model.state_dict()
        model_dict.update(part_sd)
        model.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    for ep in range(5):
        for step, batch in enumerate(train_data_loader):
            batch_x = batch[0]
            batch_y = batch[1][:,0]
            targets = {"label": batch_y}
            loss = optim_step(model, batch_x, targets, optimizer,
                    get_loss_fn("cross_entropy_loss"), params)

        # Here was originally a redundant evaluation function with randomness, and this sentence was added to ensure reproducibility.
        # In practice, this sentence can be deleted.
        rs = torch.empty((), dtype=torch.int64).random_()
        
    return model

def cvr_train(sub_model, datasets, params, wandb):
    model = get_model("MLP_SIG", params)
    models = {"model": model.to("cuda"), "submodel": sub_model.to("cuda")}

    optimizer = get_optimizer(models["model"].parameters(), params["optimizer"], params)

    train_dataset = datasets["train"]
    train_data_x = torch.from_numpy(train_dataset["x"].to_numpy().astype(np.float32))
    train_data_label = torch.from_numpy(train_dataset["labels"])
    train_data = DelayDataset(train_data_x, train_data_label)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)

    valid_dataset = datasets["valid"]
    valid_data_x = torch.from_numpy(valid_dataset["x"].to_numpy().astype(np.float32))
    valid_data_label = torch.from_numpy(valid_dataset["labels"])
    valid_data = torch.utils.data.TensorDataset(valid_data_x, valid_data_label)
    valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size=params["batch_size"])

    test_dataset = datasets["test"]
    test_data_x = torch.from_numpy(test_dataset["x"].to_numpy().astype(np.float32))
    test_data_label = torch.from_numpy(test_dataset["labels"])
    test_data = torch.utils.data.TensorDataset(test_data_x, test_data_label)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=params["batch_size"])

    data_loaders = {
        "train_data" : train_data_loader,
        "test_data" : test_data_loader,
        "valid_data" : valid_data_loader
    }
    optimizers = {
        "optimizer" : optimizer
    }


    return train(models, optimizers, data_loaders, params, wandb)


def train(models, optimizers, data_loaders, params, wandb):
    train_data = data_loaders["train_data"]
    valid_data = data_loaders["valid_data"]
    test_data = data_loaders["test_data"]
    best_model = None

    optimizer = optimizers["optimizer"]

    loss_fn = get_loss_fn(params["loss"])
    val_llloss = []
    test_auc, test_prauc, test_llloss = [], [], []

    for ep in range(params["train_epoch"]):
        vllloss = get_valid_llloss_blc(models["model"], valid_data, models["submodel"])
        print("Val ep{}, llloss {}".format(ep, vllloss))
        tauc, tprauc, tllloss = test(models["model"], test_data, params)
        print("Test ep{}, auc {}, prauc {}, llloss {}".format(ep, tauc, tprauc, tllloss))

        if len(val_llloss) == 0 or vllloss < min(val_llloss):
            best_model = models["model"].state_dict()

        val_llloss.append(vllloss)
        test_auc.append(tauc)
        test_prauc.append(tprauc)
        test_llloss.append(tllloss)
        
        if len(val_llloss) - val_llloss.index(min(val_llloss)) > params["early_stop"]:
            best_ep = val_llloss.index(min(val_llloss))
            print("Early stop at ep {}. Best ep {}. Best val_lloss {}.".format(ep, best_ep, min(val_llloss)))
            print("Final test evaluation: auc {}, prauc {}, llloss {}.".format(test_auc[best_ep], test_prauc[best_ep], test_llloss[best_ep]))
            break
        train_loss = []
        for step, batch in enumerate(tqdm(train_data)):
            batch_x = batch[0].to("cuda")
            batch_y = batch[1].to("cuda")
            targets = {"label": batch_y}

            models["model"].train()
            outputs = models["model"](batch_x)
            models["submodel"].eval()
            logits0 = models["submodel"](batch_x)["logits"]
            correction_label = torch.nn.Sigmoid()(logits0).flatten()
            targets["label"] = torch.where(targets["label"] < 1, correction_label, targets["label"].float())
            outputs = {
                "logits": outputs["logits"],
                "logits0": logits0
            }

            loss_dict = loss_fn(targets, outputs, params)
            loss = loss_dict["loss"]
            train_loss.append(loss.detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            params["log_step"] += 1
        print("Train ep{}, loss {}".format(ep, np.mean(train_loss)))
    
    models["model"].load_state_dict(best_model)
    return models["model"]

