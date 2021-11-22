from networks.srnet import Generator, Discriminator
from networks.srnet import Generator as Generator_ss
from networks.vgg19 import Vgg19
from networks.scene_text_recognition.cal_loss import * 
from tools.utils import *

def build_model(model_type, device, model_params=None, pretrained=False):
    print_log('build_model: ' + model_type)
    if model_type == 'srnet':
        G = Generator(in_channels = 3)
        D1 = Discriminator(in_channels = 6)    
        D2 = Discriminator(in_channels = 6)
        vgg_features = Vgg19()
        return G, D1, D2, vgg_features, None, None
    elif model_type == 'ss':
        G = Generator_ss(in_channels = 3)
        D1 = Discriminator(in_channels = 6)    
        D2 = Discriminator(in_channels = 6)
        vgg_features = Vgg19()
        recognition, recognition_criterion = get_model(device)
        return G, D1, D2, vgg_features, recognition, recognition_criterion
        