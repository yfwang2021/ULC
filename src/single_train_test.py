from models import get_model
from loss import get_loss_fn
from utils import get_optimizer
from metrics import cal_llloss_with_logits, cal_auc, cal_llloss_with_logits_and_weight, cal_prauc
from data import get_criteo_dataset, DelayDataset
from tqdm import tqdm
import numpy as np
import torch
from loss import stable_log1pex

def test(model, test_data, params):
    all_logits = []
    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_data)):
            batch_x = batch[0].to("cuda")
            batch_y = batch[1].to("cuda")
            logits = model(batch_x)["logits"]
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            all_probs.append(torch.nn.Sigmoid()(logits).cpu())
    all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1,))
    all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1,))
    all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1,))
    llloss = cal_llloss_with_logits(all_labels, all_logits)
    auc = cal_auc(all_labels, all_probs)
    prauc = cal_prauc(all_labels, all_probs)
    return auc, prauc, llloss

def get_valid_llloss(model, test_data, meta0_model, meta1_model, params):
    all_logits = []
    all_probs = []
    all_labels = []
    all_logits0 = []
    all_logits1 = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_data)):
            batch_x = batch[0].to("cuda")
            batch_y = batch[1].to("cuda")
            logits0 = meta0_model(batch_x)["logits"]
            logits1 = meta1_model(batch_x)["logits"]  
            logits = model(batch_x)["logits"]
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            all_probs.append(torch.nn.Sigmoid()(logits).cpu())
            all_logits0.append(torch.nn.Sigmoid()(logits0).cpu().numpy())
            all_logits1.append(torch.nn.Sigmoid()(logits1).cpu().numpy())
    all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1,))
    all_logits1 = np.reshape(np.concatenate(all_logits1, axis=0), (-1,))
    all_logits0 = np.reshape(np.concatenate(all_logits0, axis=0), (-1,))
    all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1,))
    all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1,))
 

    llloss = cal_llloss_with_logits_and_weight(all_labels, all_logits, all_logits0, all_logits1)
    return llloss

def get_valid_llloss_blc(model, test_data, meta0_model, meta1_model, params):
    all_logits = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_data)):
            batch_x = batch[0].to("cuda")
            batch_y = batch[1].to("cuda")
            logits0 = meta0_model(batch_x)["logits"]

            corrections = (torch.nn.Sigmoid()(logits0)).flatten()
            
            corrections = torch.where(batch_y < 1, corrections, batch_y.float())
            logits = model(batch_x)["logits"]
            all_logits.append(logits.cpu().numpy())
            all_labels.append(corrections.cpu().numpy())
    all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1,))
    all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1,))
 

    llloss = cal_llloss_with_logits(all_labels, all_logits)
    return llloss

def train(models, optimizers, data_loaders, params, wandb):
    train_data = data_loaders["train_data"]
    S = len(train_data.dataset)
    print("train_data_size: {}".format(S))
    valid_data = data_loaders["valid_data"]
    test_data = data_loaders["test_data"]

    optimizer = optimizers["optimizer"]

    loss_fn = get_loss_fn(params["loss"])
    val_llloss = []
    test_auc, test_prauc, test_llloss = [], [], []
    log_step = 0
    for ep in range(params["train_epoch"]):
        if params["method"] == "FSIW":
            vllloss = get_valid_llloss(models["model"], valid_data, models["fsiw0"], models["fsiw1"], params)
        elif params["method"] == "BasicLC":
            vllloss = get_valid_llloss_blc(models["model"], valid_data, models["fsiw0"], models["fsiw1"], params)
        else:
            _, _, vllloss = test(models["model"], valid_data, params)
        print("Val ep{}, llloss {}".format(ep, vllloss))
        tauc, tprauc, tllloss = test(models["model"], test_data, params)
        print("Test ep{}, auc {}, prauc {}, llloss {}".format(ep, tauc, tprauc, tllloss))

        val_llloss.append(vllloss)
        test_auc.append(tauc)
        test_prauc.append(tprauc)
        test_llloss.append(tllloss)
            
        if len(val_llloss) - val_llloss.index(min(val_llloss)) > params["early_stop"]:
            best_ep = val_llloss.index(min(val_llloss))
            print("Early stop at ep {}. Best ep {}. Best val_lloss {}".format(ep, best_ep, min(val_llloss)))
            print("Final test evaluation: auc {}, prauc {}, llloss {}.".format(test_auc[best_ep], test_prauc[best_ep], test_llloss[best_ep]))
            break

        if params["method"] not in ["nnDF"]:
            train_loss = []
            for step, batch in enumerate(tqdm(train_data)):
                batch_x = batch[0].to("cuda")
                batch_y = batch[1].to("cuda")
                targets = {"label": batch_y}

                models["model"].train()
                outputs = models["model"](batch_x)
                if params["method"] == "FSIW":
                    models["fsiw0"].eval()
                    models["fsiw1"].eval()
                    logits0 = models["fsiw0"](batch_x)["logits"]
                    logits1 = models["fsiw1"](batch_x)["logits"]
                    outputs = {
                        "logits": outputs["logits"],
                        "logits0": logits0,
                        "logits1": logits1
                    }

                loss_dict = loss_fn(targets, outputs, params)
                loss = loss_dict["loss"]
                train_loss.append(loss.detach().cpu())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                log_step += 1
            print("Train ep{}, loss {}".format(ep, np.mean(train_loss)))
        else:
            train_loss = []
            all_jd_pos = [0,0]
            all_jd_neg = [0,0]
            all_je_pos = [0,0]
            all_je_neg = [0,0]
            N = 0
            M = 0
            flag = 0
            
            with torch.no_grad():
                for step, batch in enumerate(tqdm(train_data)):
                    batch_x = batch[0].to("cuda")
                    batch_y = batch[1].to("cuda")
                    targets = {"label": batch_y}

                    models["model"].train()
                    outputs = models["model"](batch_x)

                    oy = targets["label"][:,0]
                    fn = targets["label"][:,1]
                    ine = targets["label"][:,2]
                    x = outputs["logits"]
                    pos_loss = stable_log1pex(x)
                    neg_loss = x + stable_log1pex(x)
                    all_jd_pos[0] += torch.sum(pos_loss*oy)
                    all_jd_pos[1] += pos_loss.shape[0]
                    all_je_pos[0] += torch.sum(pos_loss*fn)
                    all_je_pos[1] += torch.sum(ine)
                    all_jd_neg[0] += torch.sum(neg_loss*(1-oy))
                    all_jd_neg[1] += neg_loss.shape[0]
                    all_je_neg[0] += torch.sum(neg_loss*fn)
                    all_je_neg[1] += torch.sum(ine)

                N = all_jd_pos[1]
                M = all_je_pos[1]
                all_jd_pos = all_jd_pos[0]/all_jd_pos[1]
                all_jd_neg = all_jd_neg[0]/all_jd_neg[1]
                all_je_pos = all_je_pos[0]/all_je_pos[1]
                all_je_neg = all_je_neg[0]/all_je_neg[1]

                if all_jd_neg - all_je_neg > 0:
                    flag = 0
                else:
                    flag = 1
            
            if params["method"] == "nnDF":
                train_loss = []
                optimizer.zero_grad()
                for step, batch in enumerate(tqdm(train_data)):
                    batch_x = batch[0].to("cuda")
                    batch_y = batch[1].to("cuda")
                    targets = {"label": batch_y}

                    models["model"].train()
                    outputs = models["model"](batch_x)

                    oy = targets["label"][:,0]
                    fn = targets["label"][:,1]
                    ine = targets["label"][:,2]
                    x = outputs["logits"]
                    pos_loss = stable_log1pex(x)
                    neg_loss = x + stable_log1pex(x)
                    jd_pos = torch.sum(pos_loss*oy)/N
                    je_pos = torch.sum(pos_loss*fn)/M
                    jd_neg = torch.sum(neg_loss*(1-oy))/N
                    je_neg = torch.sum(neg_loss*fn)/M
                    if flag == 1:
                        loss = jd_pos + je_pos
                    else:
                        loss = jd_pos + je_pos + jd_neg - je_neg

                    train_loss.append(loss.detach().cpu())
                    loss.backward()
                optimizer.step()
                log_step += 1
            print("Train ep{}, loss {}".format(ep, np.mean(train_loss)))

def next_run(params, wandb):
    datasets = get_criteo_dataset(params)
    np.random.seed(params["seed"])

    if params["method"] == "DFM":
        model = get_model("MLP_EXP_DELAY", params)
    elif params["method"] == "nnDF":
        model = get_model("MLP_SIG", params)
    else:
        model = get_model("MLP_SIG", params)
    models = {"model": model.to("cuda")}
    if params["method"] == "FSIW" or params["method"] == "BasicLC":
        fsiw0_model = get_model("MLP_FSIW", params)
        fsiw0_model.load_state_dict(torch.load(params["pretrain_fsiw0_model_ckpt_path"]))
        fsiw1_model = get_model("MLP_FSIW", params)
        fsiw1_model.load_state_dict(torch.load(params["pretrain_fsiw1_model_ckpt_path"]))
        models["fsiw0"] = fsiw0_model.to("cuda")
        models["fsiw1"] = fsiw1_model.to("cuda")
    elif params["method"] == "DFM":
        dfm_model = get_model("MLP_EXP_DELAY", params)
        models["model"] = dfm_model.to("cuda")

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

    train(models, optimizers, data_loaders, params, wandb)
