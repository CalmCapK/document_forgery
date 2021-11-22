import argparse
import torch.multiprocessing as mp
import yaml

from pipelines.solver import Solver
from tools.utils import print_log

import cv2
import os
import numpy as np

from pygame import freetype

from tools import render_standard_text
from skimage.transform import resize
import torch

import torchvision.transforms.functional as F
from skimage import io

def resize_image(i_s):
    h, w = i_s.shape[:2]
    scale_ratio = 63 / h
    to_h = 63
    to_w = int(w * scale_ratio)
    to_w = int(round(to_w / 8)) * 8 - 1
    to_scale = (to_h, to_w)

    i_s = resize(i_s, to_scale, preserve_range=True)
            
    # h*w*c->c*h*w
    i_s = i_s.transpose((2, 0, 1))

    i_s_batch = []
    i_s_batch.append(i_s) 
    # batch*c*h*w
    i_s_batch = np.stack(i_s_batch)
    i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.) 
    return i_s_batch

def infer(i_s, i_t):
    global solver
    solver.G.train(False)
    solver.G.eval()
    #hxwxc
    i_t = resize_image(i_t)
    i_s = resize_image(i_s)
    #1xcxhxw
    i_t = i_t.to(solver.device)
    i_s = i_s.to(solver.device)
    o_sk, o_t, o_b, o_f = solver.G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
    return o_sk, o_t, o_b, o_f

def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 2) 
        cv2.imshow('image', img2)
        min_x = min(point1[0],point2[0])     
        min_y = min(point1[1],point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])
        
        i_s = img[min_y:min_y+height, min_x:min_x+width]
        i_t = get_text_image(i_s.shape[0], i_s.shape[1])
        o_sk, o_t, o_b, o_f = infer(i_s, i_t)
        #1xcxhxw
        #o_f = o_b
        o_f = o_f[0].cpu().detach().numpy()
        #cxhxw
        o_f = (o_f + 1)/2*255
        o_f = o_f.transpose((1, 2, 0))
        #o_f_single = F.to_pil_image((o_f[0] + 1)/2)
        #print(o_f.shape)
        o_f = resize(o_f, (i_s.shape[0], i_s.shape[1]), preserve_range=True)
        
        #print(o_f.shape)
        img_r = img.copy()
        img_r[min_y:min_y+height, min_x:min_x+width] = o_f
        img[min_y:min_y+height, min_x:min_x+width] = o_f
        img_r = cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./result.png', img_r)
        o_f = cv2.cvtColor(o_f, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./o_f.png', o_f)


def get_text_image(h=30, w=100):
    standard_font_path = '/data/kzy/SRNet/SRNet_pytorch/datasets/fonts/chinese_ttf/仿宋_GB2312.ttf'   
    text = "小 明"
    freetype.init()
    i_t = render_standard_text.make_standard_text(standard_font_path, text, (h, w))
    i_t_path = os.path.join('i_t.png')
    cv2.imwrite(i_t_path, i_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return i_t
    
    #cv2.imwrite(i_t_path, i_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

def get_source_image(image_path):
    global img
    #img = cv2.imread(image_path)
    img = io.imread(image_path)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows() 

def get_source_image2(image_path):
    #img = cv2.imread(image_path)
    img = io.imread(image_path)
          
    min_x = 0    
    min_y = 0
    width = 100#img.shape[1]
    height = 40#img.shape[0]
    i_s = img[min_y:min_y+height, min_x:min_x+width]
    cv2.imwrite('./i_s.png', img)
    i_t = get_text_image(i_s.shape[0], i_s.shape[1])
    o_sk, o_t, o_b, o_f = infer(i_s, i_t)
    #1xcxhxw
    o_f2 = F.to_pil_image((o_f[0] + 1)/2)
    o_f2.save(os.path.join('./o_f2.png'))  
    o_f = o_f[0].cpu().detach().numpy()
    #o_f = o_f.squeeze(0).to('cpu')
    #cxhxw
    o_f = (o_f + 1)/2*255
    #print(o_f.shape)
    o_f = o_f.transpose((1, 2, 0))
    #print(o_f.shape)
    #cv2.imwrite('./o_f1.png', o_f)
    o_f = resize(o_f, (i_s.shape[0], i_s.shape[1]), preserve_range=True)
    #print(o_f.shape)
    img[min_y:min_y+height, min_x:min_x+width] = o_f
    cv2.imwrite('./o_f.png', o_f)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #加这和PIL一样，但四和结果不一样
    cv2.imwrite('./result.png', img)
    


def main(local_rank, config):
    print_log("local_rank:" + str(local_rank))
    config.local_rank = local_rank
    global solver 
    solver = Solver(config)
    
   #image_path = './real_data/i_s/0.png'
    #image_path = './gongji/0.png'
    image_path = '/home/kezhiying/data_doc/zhengshu/4.png' #13 14 18
    #image_path = './srnet_data_chinese_test/i_s/1.png'
    get_source_image(image_path)
    #get_source_image2(image_path)

    

    


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='./configs/config_app.yaml')
    #Distributed_1: local_rank代表当前进程，分布式启动会会自动更新该参数,不需要在命令行显式调用
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)
    
    #get_text_image()
    #single, DataParallel, Distributed, Distributed_Apex
    main(args.local_rank, argparse.Namespace(**config))


 

 