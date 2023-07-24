from models import get_model
from loss import get_loss_fn
from utils import get_optimizer
from tqdm import tqdm
import numpy as np
import torch
from data import get_criteo_dataset

def optim_step(model, x, targets, optimizer, loss_fn, params):
    model.train()
    outputs = model(x)
    loss_dict = loss_fn(targets, outputs, params)
    loss = loss_dict["loss"]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().cpu()

def train(model, optimizer, train_data, params):
    mean_loss = []
    for step, batch in enumerate(tqdm(train_data)):
        batch_x = batch[0]
        batch_y = batch[1]
        targets = {"label": batch_y}
        loss = optim_step(model, batch_x, targets, optimizer,
                   get_loss_fn(params["loss"]), params)
        mean_loss.append(loss)
    return np.mean(mean_loss)


def run(params):
    dataset = get_criteo_dataset(params)
    train_dataset = dataset["train"]

    train_data_x = torch.from_numpy(train_dataset["x"].to_numpy().astype(np.float32)).to("cuda")
    train_data_label = torch.from_numpy(train_dataset["labels"]).to("cuda")
    train_data = torch.utils.data.TensorDataset(train_data_x, train_data_label)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)

    model = get_model(params["model"], params).to("cuda")
    optimizer = get_optimizer(model.parameters(), params["optimizer"], params)
    for ep in range(params["epoch"]):
        loss = train(model, optimizer, train_data_loader, params)
        print("ep {} train loss {}".format(ep, loss))
        torch.save(model.state_dict(), params["model_ckpt_path"])