from .resnet import *
from .wide_resnet import *
from .mobilenetv2 import *
from .model_builder import *
from .detr import *
from .backbone import *
from .position_encoding import *
from .transformer import *  
from .matcher import *  


def build_model(args):
    return build(args)