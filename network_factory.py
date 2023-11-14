from models.BasicCNN import BasicCNN
from models.BasicCNN_2 import BasicCNN_2
from models.BasicCNN_3 import BasicCNN_3
from models.ResNet50 import ResNet50


def network_factory(model_type, **model_kwargs):
    if model_type == "BasicCNN":
        return BasicCNN(**model_kwargs)
    if model_type == "BasicCNN_2":
        return BasicCNN_2(**model_kwargs)
    if model_type == "BasicCNN_3":
        return BasicCNN_3(**model_kwargs)
    if model_type == "ResNet50":
        return ResNet50(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")