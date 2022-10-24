from src.models.multi_layer_perception import MultiLayerPerception
from src.data.numerical_paired_dataset import NumericalPairedDataset
import torch
import torch.nn as nn
import numpy as np
from src.trainer import Trainer
import src.utils as utils
import logging
import os
import sys


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


logger = logging.getLogger("train")

class Config():    
    # common parameters
    seed = 1
    cpu = True
    batch_size = 32
    valid_batch_size = batch_size

    # train parameters
    learning_rate = 1e-2
    max_epoch = 100
    patience = 20

    # dataset parameters for function fitting
    samples = 1000
    variables = 1
    scale = 4 * np.pi
    split_ratio = 0.1

    # model parameters
    num_layers = 4
    input_dim = variables
    output_dim = 1
    hidden_size = 32
    dropout = 0
    activation_fn = 'sigmoid'


    residual_connection = False
    batch_norm = False


def train(cfg, trainer):
    itr = trainer.get_train_iterator()

    for epoch in range(cfg.max_epoch):
        train_loss = 0
        for i, sample in enumerate(itr):
            loss = trainer.train_step(sample)
            train_loss += loss.item()
        valid_loss = validate(cfg, trainer)

        logger.info("end of epoch {}, train loss {:.4f}, valid loss {:.4f}".format(epoch + 1, train_loss, valid_loss))

        should_stop = should_stop_early(cfg, valid_loss)
        if should_stop:
            break
    logger.info("finish training at epoch {}".format(epoch + 1))

def validate(cfg, trainer):
    itr = trainer.get_valid_iterator()
    valid_loss = 0
    for i, sample in enumerate(itr):
        valid_loss += trainer.valid_step(sample)
    return valid_loss


def should_stop_early(cfg, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.patience <= 0:
        return False

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or valid_loss < prev_best:
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.patience
                )
            )
            return True
        else:
            return False

def evaluate(cfg, model, criterion, test_set):
    logger = logging.getLogger("evaluate")
    logger.info("start evaluation")
    from torch.utils.data import DataLoader
    model.eval()
    criterion.eval()

    logger.info("loading {} testing samples".format(len(test_set)))
    itr = DataLoader(test_set, batch_size=cfg.batch_size, collate_fn=test_set.collater, shuffle=True)
    test_loss = 0
    with torch.no_grad():
        for i, sample in enumerate(itr):
            net_output = model(**sample["net_input"])
            loss = criterion(net_output, sample["target"])
            test_loss += loss.item() / sample["nsamples"]
    logger.info("finish evaluation, test loss {:.4f}".format(test_loss))
    return test_loss

def visualization(model, test_set):

    x, y = test_set.src, test_set.tgt
    y_pred = model(test_set.src)

    x = x.detach().numpy().flatten()
    y = y.detach().numpy().flatten()
    y_pred = y_pred.detach().numpy().flatten()

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(x, y, label = "ground truth")
    ax.scatter(x, y_pred, label = "prediction")
    ax.legend()
    ax.set_title("Function Fitting")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


def create_dataset(cfg):
    x = torch.rand([cfg.samples, cfg.variables], dtype=torch.float32, device="cuda" if not cfg.cpu else "cpu") * cfg.scale
    y = torch.sin(x) + torch.exp(-x)

    split = int(cfg.split_ratio * cfg.samples)
    valid_set = NumericalPairedDataset(x[:split], y[:split])
    test_set = NumericalPairedDataset(x[split: 2*split], y[split: 2*split])
    train_set = NumericalPairedDataset(x[2*split:], y[2*split:])
    return train_set, valid_set, test_set

def main():
    cfg = Config()

    np.random.seed(cfg.seed)
    utils.set_torch_seed(cfg.seed)

    train_set, valid_set, test_set = create_dataset(cfg)
    logger.info("loading {} training samples and {} validating samples".format(len(train_set), len(valid_set)))

    model = MultiLayerPerception(cfg)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    trainer = Trainer(cfg, model, criterion, optimizer, train_set, valid_set)

    logger.info("start training")
    train(cfg, trainer)
    
    evaluate(cfg, model, criterion, test_set)

    visualization(model, test_set)


if __name__ == '__main__':
    main()


