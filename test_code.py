import torch
from tools.eval import *
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from skimage import io
import cv2
from shutil import copyfile
result_paths = [
    'exp1_20211027033218_srnet_epoch_436452_best',
    'exp1_20211027033218_srnet_epoch_500000',
    'exp1_20211027033218_srnet_epoch_450000',
    'exp1_20211027033218_srnet_epoch_400000',
    'exp1_20211027033218_srnet_epoch_350000',
    'exp1_20211027033218_srnet_epoch_315916_best',
    'exp1_20211027033218_srnet_epoch_295828_best',
    'exp1_20211027033218_srnet_epoch_250000',
    'exp1_20211027033218_srnet_epoch_201116_best',
    'exp1_20211027033218_srnet_epoch_165008_best',
    'exp1_20211027033218_srnet_epoch_100000',

    'exp2_20211027033432_srnet_epoch_320000',
    'exp2_20211027033432_srnet_epoch_315916_best',
    'exp2_20211027033432_srnet_epoch_298966_best',
    'exp2_20211027033432_srnet_epoch_200000',
    'exp2_20211027033432_srnet_epoch_168254_best',
    'exp2_20211027033432_srnet_epoch_100000',
    'exp2_20211027033432_srnet_epoch_42364_best',

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
    
    'exp7_20211027095759_srnet_epoch_500000',
    'exp7_20211027095759_srnet_epoch_436452_best',
    'exp7_20211027095759_srnet_epoch_370000',
    'exp7_20211027095759_srnet_epoch_323154_best',
    'exp7_20211027095759_srnet_epoch_270000',
    'exp7_20211027095759_srnet_epoch_227058_best',
    'exp7_20211027095759_srnet_epoch_168254_best',
    'exp7_20211027095759_srnet_epoch_100000',
    'exp7_20211027095759_srnet_epoch_42364_best',
    'exp7_20211027095759_srnet_epoch_22910_best',
    'exp7_20211027095759_srnet_epoch_9266_best',
    'exp7_20211027095759_srnet_epoch_1256_best',
    'exp7_20211027095759_srnet_epoch_788_best',
    'exp7_20211027095759_srnet_epoch_118_best',
    'exp7_20211027095759_srnet_epoch_46_best',
    'exp7_20211027095759_srnet_epoch_2_best',
    
]

result_path_chinese2 = [
    'chinese2_exp1_20211108031308_srnet_epoch_495000',
    'chinese2_exp1_20211108031308_srnet_epoch_450000',
    'chinese2_exp1_20211108031308_srnet_epoch_400000',
    'chinese2_exp1_20211108031308_srnet_epoch_380398_best',
    'chinese2_exp1_20211108031308_srnet_epoch_350000',
    'chinese2_exp1_20211108031308_srnet_epoch_302764_best',
    'chinese2_exp1_20211108031308_srnet_epoch_268346_best',
    'chinese2_exp1_20211108031308_srnet_epoch_200000',
    'chinese2_exp1_20211108031308_srnet_epoch_187724_best',
    'chinese2_exp1_20211108031308_srnet_epoch_150000',
    'chinese2_exp1_20211108031308_srnet_epoch_100000',
    'chinese2_exp1_20211108031308_srnet_epoch_000002_best',
]

result_path_chinese3 = [
    'chinese3_exp1_20211108184743_srnet_epoch_500000',
    'chinese3_exp1_20211108184743_srnet_epoch_450000',
    'chinese3_exp1_20211108184743_srnet_epoch_400000',
    'chinese3_exp1_20211108184743_srnet_epoch_351338_best',
    'chinese3_exp1_20211108184743_srnet_epoch_288876_best',
    'chinese3_exp1_20211108184743_srnet_epoch_250000',
    'chinese3_exp1_20211108184743_srnet_epoch_221874_best',
    'chinese3_exp1_20211108184743_srnet_epoch_150000',
    'chinese3_exp1_20211108184743_srnet_epoch_100000',
    'chinese3_exp1_20211108184743_srnet_epoch_050000',
    'chinese3_exp1_20211108184743_srnet_epoch_010634_best',
    'chinese3_exp1_20211108184743_srnet_epoch_000002_best',
]

#https://zodiac911.github.io/blog/matplotlib-axis.html
def vis():
    result_paths = result_path_chinese2 
    for i in range(len(result_paths)):
        result_path = '../chinese2_exp1/' + result_paths[i] + '/test/valid_samples/iter-1'
        sample_lines = os.listdir(result_path)
        fig = plt.figure(figsize=(int(600*7*1.2/100), int(70*10*1.2/100)))
        plt.suptitle(result_paths[i],fontsize=25)
        plt.subplots_adjust(wspace = 0.1, hspace =0.1)#调整子图间距
        for j in range(len(sample_lines)):
            sample_path = result_path + '/' + sample_lines[j]
            k = j % 10
            i_s = io.imread(sample_path+'/'+'i_s.png')
            i_t = io.imread(sample_path+'/'+'i_t.png')
            o_sk = io.imread(sample_path+'/'+'o_sk.png', as_gray = True)
            o_t = io.imread(sample_path+'/'+'o_t.png')
            o_b = io.imread(sample_path+'/'+'o_b.png')
            o_f = io.imread(sample_path+'/'+'o_f.png')
            t_f = io.imread(sample_path+'/'+'t_f.png')
            #i_t = cv2.imread(sample_path+'/'+'i_s.png')
            #i_s = cv2.imread(sample_path+'/'+'i_s.png')
            #o_sk = cv2.imread(sample_path+'/'+'o_sk.png')
            #o_t = cv2.imread(sample_path+'/'+'o_t.png')
            #o_b = cv2.imread(sample_path+'/'+'o_b.png')
            #o_f = cv2.imread(sample_path+'/'+'o_f.png')
            plt.subplot(10, 7, k*7+1)
            plt.imshow(i_t)
            plt.axis('off')
            plt.subplot(10, 7, k*7+2)
            plt.imshow(i_s)
            plt.axis('off')
            plt.subplot(10, 7, k*7+3)
            plt.imshow(t_f)
            plt.axis('off')
            plt.subplot(10, 7, k*7+4)
            plt.imshow(o_f)
            plt.axis('off')
            plt.subplot(10, 7, k*7+5)
            plt.imshow(o_t)
            plt.axis('off')
            plt.subplot(10, 7, k*7+6)
            plt.imshow(o_b)
            plt.axis('off')
            plt.subplot(10, 7, k*7+7)
            plt.imshow(o_sk)
            plt.axis('off')
            plt.show()
            #plt.savefig('./label_result_png/' + result_paths[i] + '.png')       
            if (k+1) % 10 == 0:
                plt.savefig('./chinese2_exp1_png/' + result_paths[i] + str(j+1) + '.png', bbox_inches='tight', pad_inches=0.1)       

            #plt.savefig('./label_result_png/' + result_paths[i] + '.png', bbox_inches='tight', pad_inches=0.1)       
#vis()
def create_data():
    path = './gongji/'
    newpath = './gongji/'
    names = os.listdir(path)
    print(len(names))
    
    for i in range(len(names)):
        print(i)
        from PIL import Image
        im = Image.open(path+names[i][:-4]+'.PNG').convert("RGB")
        im.save(newpath+str(i)+'.png')
        #copyfile(path+'/i_s/'+names[i][:-4]+'.PNG', newpath+'/i_s/'+str(i)+'.png')
        #copyfile(path+'/i_t/'+names[i][:-4]+'.png', newpath+'/i_t/'+str(i)+'.png')
    #path2 = '/home/kezhiying/data_doc/srnet_data_chinese_test_withlabel_20ch'
    #i_s_path = path2 + '/i_s'
    #names2 = os.listdir(i_s_path)
    #print(len(names2))
    #for lt in lts:
    #    for i in range(len(names2)):
    #        print(len(names)+i)
    #        copyfile(path2+'/'+lt+'/'+names2[i], newpath+'/'+lt+'/'+str(len(names)+i)+'.png')
    with open(newpath+'name.txt', "w", encoding="utf8") as f:
        for i in range(len(names)):
            f.writelines([str(i)+'.png', "\n"])



def merge_data():
    path = '/home/kezhiying/srnet_data/srnet_data/'
    path2 = '/home/kezhiying/srnet_data/srnet_data_short/'
    path3 = '/home/kezhiying/srnet_data/srnet_data_long/'
    newpath = '/home/kezhiying/srnet_data/srnet_data_chinese_test_withlabel/'
    subdirs = ['i_s', 'i_t', 'mask_t', 't_b', 't_f', 't_sk', 't_t']
    
    names = os.listdir(path+'i_s')
    print(len(names))
    for i in range(len(names)):
        print(i)
        for subdir in subdirs:
            copyfile(path+subdir+'/'+names[i][:-4]+'.png', newpath+subdir+'/'+str(i)+'.png')
    names2 = os.listdir(path2+'i_s')
    print(len(names2))
    for i in range(len(names2)):
        print(i)
        for subdir in subdirs:
            copyfile(path2+subdir+'/'+names2[i][:-4]+'.png', newpath+subdir+'/'+str(len(names)+i)+'.png')
    names3 = os.listdir(path3+'i_s')
    print(len(names3))
    for i in range(len(names3)):
        print(i)
        for subdir in subdirs:
            copyfile(path3+subdir+'/'+names3[i][:-4]+'.png', newpath+subdir+'/'+str(len(names)+len(names2)+i)+'.png')
    
    with open(newpath+'name.txt', "w", encoding="utf8") as f:
        for i in range(len(names)+len(names2)+len(names3)):
            f.writelines([str(i)+'.png', "\n"])

if __name__ == '__main__':
    #main()
    #create_data()
    #merge_data()
    vis()