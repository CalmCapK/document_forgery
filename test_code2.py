import torch
from tools.eval import *
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from skimage import io
import cv2
result_paths2 = [
    #'exp1_20211027033218_srnet_epoch_436452_best',
    #'exp1_20211027033218_srnet_epoch_500000',
    #'exp1_20211027033218_srnet_epoch_450000',
    #'exp1_20211027033218_srnet_epoch_400000',
    #'exp1_20211027033218_srnet_epoch_350000',
    #'exp1_20211027033218_srnet_epoch_315916_best',
    #'exp1_20211027033218_srnet_epoch_295828_best',
    #'exp1_20211027033218_srnet_epoch_250000',
    #'exp1_20211027033218_srnet_epoch_201116_best',
    #'exp1_20211027033218_srnet_epoch_165008_best',
    #'exp1_20211027033218_srnet_epoch_100000',

    #'exp2_20211027033432_srnet_epoch_320000',
    #'exp2_20211027033432_srnet_epoch_315916_best',
    #'exp2_20211027033432_srnet_epoch_298966_best',
    #'exp2_20211027033432_srnet_epoch_200000',
    #'exp2_20211027033432_srnet_epoch_168254_best',
    #'exp2_20211027033432_srnet_epoch_100000',
    #'exp2_20211027033432_srnet_epoch_42364_best',

    'exp3_20211027033822_srnet_epoch_500000',
    'exp3_20211027033822_srnet_epoch_450000',
    'exp3_20211027033822_srnet_epoch_436452_best',
    'exp3_20211027033822_srnet_epoch_400000',
    'exp3_20211027033822_srnet_epoch_350000',
    'exp3_20211027033822_srnet_epoch_309260_best',
    'exp3_20211027033822_srnet_epoch_250000',
    'exp3_20211027033822_srnet_epoch_201116_best',
    'exp3_20211027033822_srnet_epoch_164870_best',
    'exp3_20211027033822_srnet_epoch_111536_best',
    'exp3_20211027033822_srnet_epoch_42364_best',

    'exp4_20211027033853_srnet_epoch_500000',
    'exp4_20211027033853_srnet_epoch_453386_best',
    'exp4_20211027033853_srnet_epoch_436452_best',  
    'exp4_20211027033853_srnet_epoch_400000',
    'exp4_20211027033853_srnet_epoch_350000',
    'exp4_20211027033853_srnet_epoch_315916_best',
    'exp4_20211027033853_srnet_epoch_234566_best',
    'exp4_20211027033853_srnet_epoch_168254_best',
    'exp4_20211027033853_srnet_epoch_114844_best',
    'exp4_20211027033853_srnet_epoch_81644_best',
    'exp4_20211027033853_srnet_epoch_33158_best',


    'exp5_20211027073603_srnet_epoch_436452_best',
    'exp5_20211027073603_srnet_epoch_500000',
    'exp5_20211027073603_srnet_epoch_450000',
    'exp5_20211027073603_srnet_epoch_400000',
    'exp5_20211027073603_srnet_epoch_350000',
    'exp5_20211027073603_srnet_epoch_315916_best',
    'exp5_20211027073603_srnet_epoch_295828_best',
    'exp5_20211027073603_srnet_epoch_250000',
    'exp5_20211027073603_srnet_epoch_201116_best',
    'exp5_20211027073603_srnet_epoch_165008_best',
    'exp5_20211027073603_srnet_epoch_100000',
    'exp5_20211027073603_srnet_epoch_42364_best',
    'exp5_20211027073603_srnet_epoch_9266_best',
    'exp5_20211027073603_srnet_epoch_1256_best',
    'exp5_20211027073603_srnet_epoch_788_best',
    'exp5_20211027073603_srnet_epoch_138_best',
    'exp5_20211027073603_srnet_epoch_16_best',
    'exp5_20211027073603_srnet_epoch_2_best',
    
    #'exp7_20211027095759_srnet_epoch_500000',
    #'exp7_20211027095759_srnet_epoch_436452_best',
    #'exp7_20211027095759_srnet_epoch_370000',
    #'exp7_20211027095759_srnet_epoch_323154_best',
    #'exp7_20211027095759_srnet_epoch_270000',
    #'exp7_20211027095759_srnet_epoch_227058_best',
    #'exp7_20211027095759_srnet_epoch_168254_best',
    #'exp7_20211027095759_srnet_epoch_100000',
    #'exp7_20211027095759_srnet_epoch_42364_best',
    #'exp7_20211027095759_srnet_epoch_22910_best',
    #'exp7_20211027095759_srnet_epoch_9266_best',
    #'exp7_20211027095759_srnet_epoch_1256_best',
    #'exp7_20211027095759_srnet_epoch_788_best',
    #'exp7_20211027095759_srnet_epoch_118_best',
    #'exp7_20211027095759_srnet_epoch_46_best',
    #'exp7_20211027095759_srnet_epoch_2_best',
    
]


#https://zodiac911.github.io/blog/matplotlib-axis.html
result_paths = ['./tmp']
import torchvision.transforms.functional as F
def vis():
    for i in range(len(result_paths)):
        result_path = result_paths[i] + '/test/test_samples/iter-1'
        print(result_path)
        #result_path = '../exp/' + result_paths[i] + '/test/valid_samples/iter-1'
        sample_lines = os.listdir(result_path)
        #fig = plt.figure(figsize=(int(135*8/100), int(64*8*1.2/100)))
        #plt.suptitle(result_paths[i],fontsize=25)
        #plt.subplots_adjust(wspace = 0.1, hspace =0.1)#调整子图间距
        for j in range(len(sample_lines)):
            sample_path = result_path + '/' + sample_lines[j]
            i_s = io.imread(sample_path+'/'+'i_s.png')
            i_t = io.imread(sample_path+'/'+'i_t.png')
            o_sk = io.imread(sample_path+'/'+'o_sk.png', as_gray = True)
            o_t = io.imread(sample_path+'/'+'o_t.png')
            o_b = io.imread(sample_path+'/'+'o_b.png')
            o_f = io.imread(sample_path+'/'+'o_f.png')
            t_f = io.imread(sample_path+'/'+'t_f.png')
            print(i_s.max())
            diff = i_s
            diff[abs(i_s - o_b) < 10] = 127
            print('---')
            #diff = F.to_pil_image((diff+1)/2)
            from PIL import Image
            im = Image.fromarray(diff)
            im.save(sample_path+'/'+'diff.png')

            #i_t = cv2.imread(sample_path+'/'+'i_s.png')
            #i_s = cv2.imread(sample_path+'/'+'i_s.png')
            #o_sk = cv2.imread(sample_path+'/'+'o_sk.png')
            #o_t = cv2.imread(sample_path+'/'+'o_t.png')
            #o_b = cv2.imread(sample_path+'/'+'o_b.png')
            #o_f = cv2.imread(sample_path+'/'+'o_f.png')
            #plt.subplot(len(sample_lines), 6, j*6+1)
            '''
            plt.subplot(len(sample_lines), 6, j*6+2)
            plt.imshow(i_s)
            plt.axis('off')
            plt.subplot(len(sample_lines), 6, j*6+3)
            plt.imshow(o_f)
            plt.axis('off')
            plt.subplot(len(sample_lines), 6, j*6+4)
            plt.imshow(o_t)
            plt.axis('off')
            plt.subplot(len(sample_lines), 6, j*6+5)
            plt.imshow(o_b)
            plt.axis('off')
            plt.subplot(len(sample_lines), 6, j*6+6)
            plt.imshow(o_sk)
            plt.axis('off')
            plt.show()
            #plt.savefig('./label_result_png/' + result_paths[i] + '.png')       
            plt.savefig('./label_result_png/' + result_paths[i] + '.png', bbox_inches='tight', pad_inches=0.1)       

            #plt.savefig('./label_result_png/' + result_paths[i] + '.png', bbox_inches='tight', pad_inches=0.1)       
            '''
vis()
