from colorama import Fore, Back, Style
import numpy as np
from skimage.transform import resize
from torch.cuda.amp import autocast as autocast
from torchvision.utils import make_grid
from tqdm import tqdm

from networks.losses import *
from tools.utils import requires_grad
from tools.eval import cal_metric

def test_step(current_step, data_loader, G, device, config, summary_writer):
    G.train(False)
    G.eval()
    results = []
    with torch.no_grad():
        for data in tqdm(data_loader, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.WHITE, Fore.RESET)): 
            #i_t, i_s, t_sk, t_t, t_b, t_f, mask_t, img_name = data
            i_t = data[0]
            i_s = data[1]
            img_name = data[-1]
            i_t = i_t.to(device)
            i_s = i_s.to(device)
            o_sk, o_t, o_b, o_f = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
            #print(i_t[0].max())
            #print(i_t[0].min())
            #summary_writer.add_image(str(1), make_grid(i_t.detach().cpu(), nrow=8, padding=2, normalize=True, pad_value=1), 1)
            #for i in range(8):
            #    summary_writer.add_image(str(i+2), make_grid(i_t[i].detach().cpu(), nrow=3, padding=2, normalize=True, pad_value=1), 1)
            #for i in range(8):
            #    summary_writer.add_image(str(i+2), make_grid(i_t[i].detach().cpu().unsqueeze(dim=1), nrow=3, padding=2, normalize=False, pad_value=1), 2)
            #参考：https://zhuanlan.zhihu.com/p/60753993?utm_source=wechat_session&utm_medium=social&utm_oi=629892201212153856
            
            #把维度为0的删掉，降维
            #i_t = i_t.squeeze(0).to('cpu')
            #i_s = i_s.squeeze(0).to('cpu')
            #o_sk = o_sk.squeeze(0).to('cpu')
            #o_t = o_t.squeeze(0).to('cpu')
            #o_b = o_b.squeeze(0).to('cpu')
            #o_f = o_f.squeeze(0).to('cpu')
            i_t = i_t.to('cpu')
            i_s = i_s.to('cpu')
            o_sk = o_sk.to('cpu')
            o_t = o_t.to('cpu')
            o_b = o_b.to('cpu')
            o_f = o_f.to('cpu')
    
            results.append([i_t, i_s, o_sk, o_t, o_b, o_f, None, img_name])
    return results

def test_step_with_label(current_step, data_loader, G, device, config, summary_writer):
    G.train(False)
    G.eval()
    results = []
    metrics_list = None
    with torch.no_grad():
        for data in tqdm(data_loader, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.WHITE, Fore.RESET)): 
            #i_t, i_s, t_sk, t_t, t_b, t_f, mask_t, img_name = data
            i_t = data[0]
            i_s = data[1]
            t_f = data[2]
            img_name = data[-1]
            i_t = i_t.to(device)
            i_s = i_s.to(device)
            t_f = t_f.to(device)
           
            o_sk, o_t, o_b, o_f = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))

            #o_f = torch.nn.functional.interpolate(o_f, size=[i_s.shape[2], i_s.shape[3]], mode='nearest')
            metric = cal_metric(t_f, o_f)
            if metrics_list is None:
                metrics_list = metric
            else:
                metrics_list = [i + j for i, j in zip(metrics_list, metric)]
            #把维度为0的删掉，降维
            #i_t = i_t.squeeze(0).to('cpu')
            #i_s = i_s.squeeze(0).to('cpu')
            #o_sk = o_sk.squeeze(0).to('cpu')
            #o_t = o_t.squeeze(0).to('cpu')
            #o_b = o_b.squeeze(0).to('cpu')
            #o_f = o_f.squeeze(0).to('cpu')
            #t_f = t_f.squeeze(0).to('cpu')

            i_t = i_t.to('cpu')
            i_s = i_s.to('cpu')
            o_sk = o_sk.to('cpu')
            o_t = o_t.to('cpu')
            o_b = o_b.to('cpu')
            o_f = o_f.to('cpu')
            t_f = t_f.to('cpu')
            results.append([i_t, i_s, o_sk, o_t, o_b, o_f, t_f, img_name])
    return results, np.mean(metrics_list, axis=1)