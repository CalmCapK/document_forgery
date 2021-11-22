import horovod.torch as hvd
import numpy as np
from numpy.core.fromnumeric import reshape
import os
from skimage import io
from skimage.transform import resize
import torch
from torch.utils import data

from tools import utils

class custom_collate():
    def __init__(self, mode, data_type, data_shape):
        self.mode = mode
        self.data_type = data_type
        self.data_shape = data_shape
    
    def process_with_label(self, batch):
        i_t_batch, i_s_batch = [], []
        t_sk_batch, t_t_batch, t_b_batch, t_f_batch = [], [], [], []
        mask_t_batch = []
        img_name_batch = []

        w_sum = 0
        for item in batch:
            i_s = item[1]
            h, w = i_s.shape[:2]
            scale_ratio = self.data_shape[0] / h 
            w_sum += int(w * scale_ratio)

        to_h = self.data_shape[0]
        to_w = w_sum // len(batch)
        to_w = int(round(to_w / 8)) * 8 - 1
        to_scale = (to_h, to_w)

        for item in batch:
            i_t, i_s, t_sk, t_t, t_b, t_f, mask_t, img_name = item

            i_t = resize(i_t, to_scale, preserve_range=True)
            i_s = resize(i_s, to_scale, preserve_range=True)
            # h*w -> h*w*1
            t_sk = np.expand_dims(resize(t_sk, to_scale, preserve_range=True), axis = -1) 
            t_t = resize(t_t, to_scale, preserve_range=True)
            t_b = resize(t_b, to_scale, preserve_range=True)  
            t_f = resize(t_f, to_scale, preserve_range=True)
            mask_t = np.expand_dims(resize(mask_t, to_scale, preserve_range=True), axis = -1)

            # h*w*c->c*h*w
            i_t = i_t.transpose((2, 0, 1))
            i_s = i_s.transpose((2, 0, 1))
            t_sk = t_sk.transpose((2, 0, 1))
            t_t = t_t.transpose((2, 0, 1))
            t_b = t_b.transpose((2, 0, 1))
            t_f = t_f.transpose((2, 0, 1))
            mask_t = mask_t.transpose((2, 0, 1)) 

            i_t_batch.append(i_t) 
            i_s_batch.append(i_s)
            t_sk_batch.append(t_sk)
            t_t_batch.append(t_t) 
            t_b_batch.append(t_b) 
            t_f_batch.append(t_f)
            mask_t_batch.append(mask_t)
            img_name_batch.append(img_name)

        # batch*c*h*w
        i_t_batch = np.stack(i_t_batch)
        i_s_batch = np.stack(i_s_batch)
        t_sk_batch = np.stack(t_sk_batch)
        t_t_batch = np.stack(t_t_batch)
        t_b_batch = np.stack(t_b_batch)
        t_f_batch = np.stack(t_f_batch)
        mask_t_batch = np.stack(mask_t_batch)

        i_t_batch = torch.from_numpy(i_t_batch.astype(np.float32) / 127.5 - 1.) 
        i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.) 
        t_sk_batch = torch.from_numpy(t_sk_batch.astype(np.float32) / 255.) 
        t_t_batch = torch.from_numpy(t_t_batch.astype(np.float32) / 127.5 - 1.) 
        t_b_batch = torch.from_numpy(t_b_batch.astype(np.float32) / 127.5 - 1.) 
        t_f_batch = torch.from_numpy(t_f_batch.astype(np.float32) / 127.5 - 1.) 
        mask_t_batch =torch.from_numpy(mask_t_batch.astype(np.float32) / 255.)    
      
        return [i_t_batch, i_s_batch, t_sk_batch, t_t_batch, t_b_batch, t_f_batch, mask_t_batch, img_name_batch]

    def process_without_label(self, batch):
        i_t_batch, i_s_batch, t_f_batch = [], [], []
        img_name_batch = []

        w_sum = 0
        for item in batch:
            i_s = item[1]
            h, w = i_s.shape[:2]
            scale_ratio = self.data_shape[0] / h
            w_sum += int(w * scale_ratio)

        to_h = self.data_shape[0]
        to_w = w_sum // len(batch)
        to_w = int(round(to_w / 8)) * 8 - 1
        to_scale = (to_h, to_w)

        for item in batch:
            i_t, i_s, t_f, img_name = item

            i_t = resize(i_t, to_scale, preserve_range=True)
            i_s = resize(i_s, to_scale, preserve_range=True)
            
            # h*w*c->c*h*w
            i_t = i_t.transpose((2, 0, 1))
            i_s = i_s.transpose((2, 0, 1))

            i_t_batch.append(i_t) 
            i_s_batch.append(i_s)
            img_name_batch.append(img_name)

            if t_f is not None:
                t_f = resize(t_f, to_scale, preserve_range=True)
                t_f = t_f.transpose((2, 0, 1))
                t_f_batch.append(t_f)

        # batch*c*h*w
        i_t_batch = np.stack(i_t_batch)
        i_s_batch = np.stack(i_s_batch)

        i_t_batch = torch.from_numpy(i_t_batch.astype(np.float32) / 127.5 - 1.) 
        i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.) 

        if len(t_f_batch) > 0:
            t_f_batch = np.stack(t_f_batch)
            t_f_batch = torch.from_numpy(t_f_batch.astype(np.float32) / 127.5 - 1.) 
        return [i_t_batch, i_s_batch, t_f_batch, img_name_batch]

    def process_with_text(self, batch):
        #[i_t, i_s, t_s, source_text, target_text, img_name]
        i_t_batch, i_s_batch, t_s_batch = [], [], []
        img_name_batch = []
        source_text_batch = []
        target_text_batch = []

        w_sum = 0
        for item in batch:
            i_s = item[1]
            h, w = i_s.shape[:2]
            scale_ratio = self.data_shape[0] / h
            w_sum += int(w * scale_ratio)

        to_h = self.data_shape[0]
        to_w = w_sum // len(batch)
        to_w = int(round(to_w / 8)) * 8 - 1
        to_scale = (to_h, to_w)

        for item in batch:
            i_t, i_s, t_s, source_text, target_text, img_name = item

            i_t = resize(i_t, to_scale, preserve_range=True)
            i_s = resize(i_s, to_scale, preserve_range=True)
            
            # h*w*c->c*h*w
            i_t = i_t.transpose((2, 0, 1))
            i_s = i_s.transpose((2, 0, 1))

            i_t_batch.append(i_t) 
            i_s_batch.append(i_s)
            img_name_batch.append(img_name)

            if t_s is not None:
                t_s = resize(t_s, to_scale, preserve_range=True)
                t_s = t_s.transpose((2, 0, 1))
                t_s_batch.append(t_f)

        # batch*c*h*w
        i_t_batch = np.stack(i_t_batch)
        i_s_batch = np.stack(i_s_batch)

        i_t_batch = torch.from_numpy(i_t_batch.astype(np.float32) / 127.5 - 1.) 
        i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.) 

        if len(t_s_batch) > 0:
            t_s_batch = np.stack(t_s_batch)
            t_s_batch = torch.from_numpy(t_s_batch.astype(np.float32) / 127.5 - 1.) 
        return [i_t_batch, i_s_batch, t_s_batch, source_text, target_text, img_name_batch]

    def __call__(self, batch):
        '''在这里重写collate_fn函数'''
        if self.data_type == 'data_srnet':
            if self.mode == 'train':
                return self.process_with_label(batch)
            else:
                return self.process_without_label(batch)
        if self.data_type == 'data_imgur5k':
            return self.process_with_text(batch)
        
        


class DataFolder(data.Dataset):
    def __init__(self, mode='train', data_type = 'data_srnet', dataset_params=None):
        self.mode = mode
        self.data_type = data_type
        self.cfg = dataset_params
        self.data_list = utils.read_data(self.cfg['data_list_path'])
        self.data_num = len(self.data_list)
        utils.print_log("image count in {} path: {}".format(self.mode, self.data_num))

    def __getitem__(self, index):
        if self.data_type == 'data_srnet':
            return self.get_data_srnet(index)
        if self.data_type == 'data_imgur5k':
            return self.get_data_imgur5k(index)
           
    def get_data_srnet(self, index)
        img_name = self.data_list[index]
        if self.mode == 'train':
            i_t = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['i_t_dir'], img_name))
            i_s = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['i_s_dir'], img_name))
            t_sk = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['t_sk_dir'], img_name), as_gray = True)
            t_t = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['t_t_dir'], img_name))
            t_b = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['t_b_dir'], img_name))
            t_f = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['t_f_dir'], img_name))
            mask_t = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['mask_t_dir'], img_name), as_gray = True)
            return [i_t, i_s, t_sk, t_t, t_b, t_f, mask_t, img_name]
        else:
            i_t = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['i_t_dir'], img_name))
            i_s = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['i_s_dir'], img_name))
            if self.cfg.get('t_f_dir') and self.cfg.get('t_f_dir') != '':
                t_f = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['t_f_dir'], img_name))
            else:
                t_f = None
            return [i_t, i_s, t_f, img_name]

    def get_data_imgur5k(self, index)
        img_name, source_text, target_text = self.data_list[index]
        if self.mode == 'train':
            i_t = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['i_t_dir'], img_name))
            i_s = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['i_s_dir'], img_name))
            t_s = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['t_s_dir'], img_name))
            return [i_t, i_s, t_s, source_text, target_text, img_name]
        else:
            i_t = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['i_t_dir'], img_name))
            i_s = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['i_s_dir'], img_name))
            t_s = io.imread(os.path.join(self.cfg['data_dir'], self.cfg['t_s_dir'], img_name))
            return [i_t, i_s, t_s, source_text, target_text, img_name]

    def __len__(self):
        return self.data_num


def get_loader(batch_size, shuffle=True, num_workers=1,
               mode='train', data_type = 'data_srnet', dataset_params=None, parallel_type='', data_shape=None):
    dataset = DataFolder(mode=mode, data_type = data_type, dataset_params=dataset_params)
    if parallel_type == 'Distributed' or parallel_type == 'Distributed_Apex':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  collate_fn=custom_collate(mode, data_type, data_shape),
                                  pin_memory = True,
                                  sampler=sampler)
    elif parallel_type == 'Horovod':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=hvd.size(),
                                                                    rank=hvd.rank())
        data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  collate_fn=custom_collate(mode, data_type, data_shape),
                                  pin_memory = True,
                                  sampler=sampler)
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=custom_collate(mode, data_type, data_shape),
                                  pin_memory = True,
                                  num_workers=num_workers)
    return data_loader