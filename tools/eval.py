# old version
#from skimage.measure import compare_ssim, compare_psnr, compare_mse
import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import torch

from geometry_score import gs

'''
官方psnr和自定义psnr
    官方和自定义psnr分别在单通道和多通道几乎没区别，在[0,255]上一致，在[-1,1] 和[0,1]有一点点区别
    单通道多通道有差别！！！
    精度上差别不大
    单batch和整个batch计算有区别

官方mse和自定义mse
    官方和自定义mse几乎没区别，在[0,255]上一致，在[-1,1] 和[0,1]有一点点区别
    单通道多通道几乎没区别
    精度上有很大区别！！！
    单batch和整个batch计算都一样

官方mssim
    单通道多通道一致
    精度上有较大区别！！！
'''

def cal_mse(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    return mse

def cal_psnr(img1, img2, data_range=255):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    if mse < 1.0e-10:
        return 100 * 1.0
    return 10 * math.log10(1.0 * data_range * data_range / mse)

def cal_psnr_multichannel(img1, img2, data_range=255):
    '''
     im: HxWxC 
    '''
    if len(im1.shape) > 2:
        im1 = im1.transpose((2, 0, 1))
        im2 = im2.transpose((2, 0, 1))
        ssim = 0.0
        for i in range(im1.shape[0]):
            psnr += cal_psnr(im1[i], im2[i], data_range)
        psnr /= 3.0
    else:
        psnr = cal_psnr(im1[i], im2[i], data_range)
    return psnr

def cal_ssim_single(y_true, y_pred, data_range=255):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    #方差等价于：(np.sum(a**2))/size-np.mean(a)**2
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true) #等价：np.sqrt(((y_true - u_true) ** 2).mean()) 
    std_pred = np.sqrt(var_pred)
    c1 = (0.01 * data_range)**2
    c2 = (0.03 * data_range)**2
    sigma12 = ((y_true - u_true) * (y_pred - u_pred)).mean()
    #协方差： np.cov(y_true.reshape(-1), y_pred.reshape(-1), ddof=0)
    ssim = (2 * u_true * u_pred + c1) * (2 * sigma12 + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def cal_ssim(im1, im2, data_range=255):
    '''
     im: HxWxC 
    '''
    if len(im1.shape) > 2:
        im1 = im1.transpose((2, 0, 1))
        im2 = im2.transpose((2, 0, 1))
        ssim = 0.0
        for i in range(im1.shape[0]):
            ssim += cal_ssim_single(im1[i], im2[i], data_range)
        ssim /= 3.0
    else:
        ssim = cal_metric_single(im1[i], im2[i], data_range)
    return ssim

def cal_metric(target, output):
    '''
    BxCxHxW
    '''
    target = target.cpu().detach().numpy()
    output = output.cpu().detach().numpy()
    mse_list = []
    psnr_list = []
    ssim_list = []
    mse2_list = []
    psnr2_list = []
    ssim2_list = []
    mse3_list = []
    psnr3_list = []
    ssim3_list = []
    for i in range(target.shape[0]):
        t1 = target[i].transpose((1, 2, 0))
        t2 = (target[i].transpose((1, 2, 0))+1)/2
        t3 = ((target[i].transpose((1, 2, 0))+1)/2*255).astype(np.uint8)
        o1 = output[i].transpose((1, 2, 0))
        o2 = (output[i].transpose((1, 2, 0))+1)/2
        o3 = ((output[i].transpose((1, 2, 0))+1)/2*255).astype(np.uint8)
        mse = mean_squared_error(o1, t1)
        psnr = peak_signal_noise_ratio(o1, t1)
        ssim = structural_similarity(o1, t1, multichannel=True)
        mse2 = mean_squared_error(o2, t2)
        psnr2 = peak_signal_noise_ratio(o2, t2)
        ssim2 = structural_similarity(o2, t2, multichannel=True)
        mse3 = mean_squared_error(o3, t3)
        psnr3 = peak_signal_noise_ratio(o3, t3)
        ssim3 = structural_similarity(o3, t3, multichannel=True)
        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        mse2_list.append(mse2)  
        psnr2_list.append(psnr2)
        ssim2_list.append(ssim2)
        mse3_list.append(mse3) 
        psnr3_list.append(psnr3)
        ssim3_list.append(ssim3)
    #print(np.mean(mse_list), np.mean(psnr_list), np.mean(ssim_list), np.mean(mse2_list),np.mean(psnr2_list), np.mean(ssim2_list), np.mean(mse3_list),np.mean(psnr3_list), np.mean(ssim3_list))
    return [mse_list, psnr_list, ssim_list, mse2_list, psnr2_list, ssim2_list, mse3_list, psnr3_list, ssim3_list]

def cal_geometric_score(images):
    rlt = gs.rlts(images, L_0=32, gamma=1.0/8, i_max=100, n=100)
    mrlt = np.mean(rlt, axis=0)
    return rlt, mrlt



def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(np.dot(sigma1, sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def cal_fid_with_act(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def cal_fid_with_images(image1, image2, model):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    fid = cal_fid_with_act(act1, act2)
    return fid 

if __name__ == '__main__':
    x = np.random.rand(64, 200, 1)
    y = np.random.rand(64, 200, 1)
    x1 = x[np.where(y > 0.5)]
    print(len(np.where(y > 0.5)))
    print(x1.shape)
    x1 = np.reshape(x1, (-1, x1.shape[0]))
    print(x1.shape)
    rlt, mrlt = cal_geometric_score(x1)
    rlt2, mrlt2 = cal_geometric_score(x1)
    print(mrlt.shape)
    s = gs.geom_score(rlt, rlt2)
    print(s)
    # prepare the inception v3 model
    #from keras.applications.inception_v3 import InceptionV3
    from keras.applications.inception_v3 import preprocess_input
    from torchvision.models.inception import inception_v3
    import cv2
    from skimage import io
    from skimage.transform import resize
    model = InceptionV3(include_top=False, pooling='avg')
    path1 = './demo_image/demo_6.png'
    path2 = './demo_image/demo_1.png'
    #images1 = cv2.imread(path1).astype('float32')
    #images2 = cv2.imread(path2).astype('float32')
    images1 = io.imread(path1)
    print(images1.shape)
    batch1 = [] 
    images1 = resize(images1, (299, 299), preserve_range=True)
    #images1 = images1.transpose((2, 0, 1))
    batch1.append(images1)    
    batch1 = np.stack(batch1)
    print(batch1.shape)
    batch2 = []
    images2 = io.imread(path1)
    print(images2.shape)
    images2 = resize(images2, (299, 299), preserve_range=True)
    #images2 = images2.transpose((2, 0, 1))
    batch2.append(images2)    
    batch2 = np.stack(batch2)
    print(batch2.shape)

    batch1 = preprocess_input(batch1)
    batch2 = preprocess_input(batch2)
    print(batch1.shape)
    print(batch2.shape)
    fid = calculate_fid(model, batch1, batch2)
    print('FID : %.3f' % fid)
    print('FID_average : %.3f' % (fid / dataset_size))
    fid = calculate_fid(model, batch1, batch1)
    print('FID : %.3f' % fid)
    print('FID_average : %.3f' % (fid / dataset_size))