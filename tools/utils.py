import csv
from datetime import datetime
import numpy as np
import os
import random
import torch
import warnings

PrintColor = {
    'black': 30,
    'red': 31,
    'green': 32,
    'yellow': 33,
    'blue': 34,
    'amaranth': 35,
    'ultramarine': 36,
    'white': 37
}

PrintStyle = {
    'default': 0,
    'highlight': 1,
    'underline': 4,
    'flicker': 5,
    'inverse': 7,
    'invisible': 8
}

def init_seed(seed):
    random.seed(seed) #Python本身的随机因素
    np.random.seed(seed) #numpy随机因素
    torch.manual_seed(seed) #pytorch cpu随机因素
    torch.cuda.manual_seed(seed)  #pytorch gpu随机因素
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  #True: 每次返回的卷积算法将是确定的
    torch.backends.cudnn.benchmark = False #GPU，将其值设置为 True，就可以大大提升卷积神经网络的运行速度

    warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

def read_data(path, data_type=None):
    lines = map(str.strip, open(path).readlines())
    data = []
    if data_type == 'data_srnet':
        for line in lines:
            data.append(line)
    if data_type == 'data_imgur5k':
        for line in lines: 
            sample_path, source_text, target_text = line.split()
            data.append((sample_path, source_text, target_text))   
    else:
        for line in lines:
           if len(line.split()) == 1:
                data.append(line)
            else:
                sample_path, label = line.split()
                #sample_path, label, _, _ = line.split()
                label = int(label)
                data.append((sample_path, label))
    return data



def get_name():
    return datetime.now().strftime('%Y%m%d%H%M%S')


def print_log(s, time_style = PrintStyle['default'], time_color = PrintColor['blue'],
                content_style = PrintStyle['default'], content_color = PrintColor['yellow']):
    
    cur_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    log = '\033[{};{}m[{}]\033[0m \033[{};{}m{}\033[0m'.format \
        (time_style, time_color, cur_time, content_style, content_color, s)
    print (log)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def record_epoch(mode, epoch, total_epoch, record, record_path):
    print('\n[%s] Epoch [%d/%d]' %
          (mode, epoch, total_epoch), end='')
    for k, v in record.items():
        print(', %s: %.4f' % (k, v), end='')
    print()
    if not os.path.exists(record_path):
        with open(record_path, 'a+') as f:
            record['epoch'] = epoch
            fieldnames = [k for k, v in record.items()]
            csv_write = csv.DictWriter(f, fieldnames=fieldnames)
            csv_write.writeheader()
            csv_write.writerow(record)
    else:
        with open(record_path, 'a+') as f:
            record['epoch'] = epoch
            fieldnames = [k for k, v in record.items()]
            csv_write = csv.DictWriter(f, fieldnames=fieldnames)
            #data_row = [epoch]
            #data_row.extend(['%.4f' % (v) for k, v in record.items()])
            #csv_write.writerow(data_row)
            csv_write.writerow(record)

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_state_dict(model, config, isopt=False):
    if isopt or config.parallel_type == 'Single' or config.parallel_type == '' or config.parallel_type == 'Horovod':
        return model.state_dict()
    else:
        return model.module.state_dict()

def save_model(epoch, max_epoch, state, model_type, save_model_path, isBetter):
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    e = str(epoch).zfill(len(str(max_epoch)))
    if isBetter:
        print(save_model_path + '/' + model_type +
              "_epoch_{}_best.pth".format(e))
        torch.save(state, save_model_path + "/" +
                   model_type + "_epoch_{}_best.pth".format(e))
    else:
        print(save_model_path + '/' + model_type +
              "_epoch_{}.pth".format(e))
        torch.save(state, save_model_path + "/" +
                   model_type + "_epoch_{}.pth".format(e))


                   
def vis(tensor_val):
    i_t_single = F.to_pil_image((i_t[i] + 1)/2)          
    img_name_single = '.'.join(img_name[i].split('.')[:-1])
    check_dir(os.path.join(save_path, img_name_single))
    i_t_single.save(os.path.join(save_path, img_name_single, 'i_t.png'))        
                
