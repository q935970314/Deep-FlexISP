import os

from torch.utils import model_zoo

from classes.fc4.squeezenet.SqueezeNet import SqueezeNet

model_urls = {
    1.0: 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    1.1: 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class SqueezeNetLoader:
    def __init__(self, version: float = 1.1):
        self.__version = version
        self.__model = SqueezeNet(self.__version)

    def load(self, pretrained: bool = False) -> SqueezeNet:
        """
        Returns the specified version of SqueezeNet
        @param pretrained: if True, returns a model pre-trained on ImageNet
        """
        if pretrained:
            path_to_local = os.path.join("assets", "pretrained")
            print("\n Loading local model at: {} \n".format(path_to_local))
            os.environ['TORCH_HOME'] = path_to_local
            self.__model.load_state_dict(model_zoo.load_url(model_urls[self.__version]))
        return self.__model
