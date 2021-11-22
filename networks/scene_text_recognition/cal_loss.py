
import argparse
from PIL import Image
from skimage import io
from skimage.transform import resize
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

from networks.scene_text_recognition.utils import AttnLabelConverter
from networks.scene_text_recognition.model import Model
from networks.scene_text_recognition.dataset import *

#from utils import AttnLabelConverter
#from model import Model
#from dataset import *

def init_opt():
    opt = {  
    'workers': 4,
    'batch_size': 8,
    'batch_max_length': 25,
    'imgH': 32,
    'imgW': 100,
    'character':'0123456789abcdefghijklmnopqrstuvwxyz',
    'Transformation': 'TPS',
    'FeatureExtraction': 'ResNet',
    'SequenceModeling': 'BiLSTM',
    'Prediction': 'Attn', 
    'num_fiducial': 20, 
    'input_channel': 1,
    'output_channel': 512,
    'hidden_size': 256,
    'num_class': 38,
    'saved_model': './networks/scene_text_recognition/TPS-ResNet-BiLSTM-Attn.pth',
    }
    opt = argparse.Namespace(**opt)
    #converter = AttnLabelConverter(opt.character)
    #opt.num_class = len(converter.character)
    #print(opt.num_class)
    return opt

def get_model(device):
    opt = init_opt()
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    return model, criterion

def transform_images(path):
    opt = init_opt()
    img = Image.open(path).convert('L')
    images = [img]
    transform = ResizeNormalize((opt.imgW, opt.imgH))
    image_tensors = [transform(image) for image in images]
    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
    labels = ['avalibale']
    print(image_tensors.shape)
    return image_tensors, labels

def transform_images2(path):
    i_s = io.imread(path)
    i_s_batch = [] 
    #to_h = i_s.shape[0]
    #to_w = i_s.shape[1]
    #to_scale = (to_h, to_w)
    #i_s = resize(i_s, to_scale, preserve_range=True)
    i_s = i_s.transpose((2, 0, 1))
    i_s_batch.append(i_s)    
    i_s_batch = np.stack(i_s_batch)
    i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.) 
    labels = ['avalibale']
    return i_s_batch, labels

def text_content_loss(image_tensors, labels, device, model, criterion):
    opt = init_opt()
    model.eval()
    with torch.no_grad():
        batch_size = image_tensors.size(0)
        #print(image_tensors.shape)
        #print(labels)
        if image_tensors.size(1) > 1:
            images = []
            for i in range(batch_size):
                single_img = F.to_pil_image((image_tensors[i] + 1)/2).convert("L")
                #single_img = transforms.ToPILImage()((image_tensors[i] + 1)/2).convert('L')
                images.append(single_img)  
            transform = ResizeNormalize((opt.imgW, opt.imgH))
            image_tensors = [transform(image) for image in images]  #transforms.ToTensor() range [0, 255] -> [0.0,1.0]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        
        converter = AttnLabelConverter(opt.character)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)
        preds = model(image_tensors, text_for_pred, is_train=False)   
        preds = preds[:, :text_for_loss.shape[1] - 1, :]
        target = text_for_loss[:, 1:]  # without [GO] Symbol
        cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))
        return cost

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = './demo_image/demo_6.png'
    opt = init_opt()
    model, criterion = get_model(device)
    
    image_tensors, labels = transform_images(path)
    image_tensors = image_tensors.to(device)
    cost = text_content_loss(image_tensors, labels, device, model, criterion)
    print(cost)
    image_tensors, labels = transform_images2(path)
    image_tensors = image_tensors.to(device)
    cost = text_content_loss(image_tensors, labels, device, model, criterion)
    print(cost)


  