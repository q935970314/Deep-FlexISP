import os

import torch
from torch import Tensor

from auxiliary.settings import DEVICE
from classes.losses.AngularLoss import AngularLoss


class Model:
    def __init__(self):
        self._device = DEVICE
        self._criterion = AngularLoss(self._device)
        self._optimizer = None
        self._network = None

    def print_network(self):
        print("\n----------------------------------------------------------\n")
        print(self._network)
        print("\n----------------------------------------------------------\n")

    def log_network(self, path_to_log: str):
        open(os.path.join(path_to_log, "network.txt"), 'a+').write(str(self._network))

    def get_loss(self, pred: Tensor, label: Tensor) -> Tensor:
        return self._criterion(pred, label)

    def train_mode(self):
        self._network = self._network.train()

    def evaluation_mode(self):
        self._network = self._network.eval()

    def save(self, path_to_log: str):
        torch.save(self._network.state_dict(), os.path.join(path_to_log, "model.pth"))

    def load(self, path_to_pretrained: str):
        path_to_model = os.path.join(path_to_pretrained, "model.pth")
        self._network.load_state_dict(torch.load(path_to_model, map_location=self._device))

    def set_optimizer(self, learning_rate: float, optimizer_type: str = "adam"):
        optimizers_map = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop}
        self._optimizer = optimizers_map[optimizer_type](self._network.parameters(), lr=learning_rate)
