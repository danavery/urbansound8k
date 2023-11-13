from BasicCNN import BasicCNN
from BasicCNN_2 import BasicCNN_2
from BasicCNN_3 import BasicCNN_3

def network_factory(model_type, **model_kwargs):
    if model_type == "BasicCNN":
        return BasicCNN(**model_kwargs)
    if model_type == "BasicCNN_2":
        return BasicCNN_2(**model_kwargs)
    if model_type == "BasicCNN_3":
        return BasicCNN_3(**model_kwargs)
