import os
from tensorboardX import SummaryWriter
import time
import torch
from torch.cuda.amp import GradScaler
import torchvision.transforms.functional as F
from tqdm import tqdm

from dataset.data_loader import get_loader
from networks.networks import build_model
from networks.opt import get_optimizer 
from pipelines.train import train_G_step, train_D_step
from pipelines.test import test_step, test_step_with_label
from tools.utils import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class Solver(object):
    def __init__(self, config):
        self.solver_init_seed(config)
        self.nprocs = self.solver_init_process_batchsize(config)
        self.device = self.solver_init_gpu(config)
        self.train_loader, self.valid_loader, self.test_loader = self.get_data_loaders(
            config)
        self.G, self.D1, self.D2, self.vgg_features, self.recognition, self.recognition_criterion = build_model(config.model_type, self.device)
        self.G_optimizer, self.D1_optimizer, self.D2_optimizer = self.solver_get_optimizer(config)
        self.G.to(self.device)
        self.D1.to(self.device)
        self.D2.to(self.device)
        self.vgg_features.to(self.device)
        self.best_g_loss = torch.tensor(999999.0)
        if config.is_load_weight:
            self.load_weights(config)
        
        self.init_parallel(config)
        if config.use_amp:
            self.scalar = GradScaler()
        else:
            self.scalar = None
        # save
        record_name = get_name() if config.save['record_name'] == '' else config.save['record_name']
        check_dir(config.save['result_path'] + '/' + record_name)
        
        for k, v in config.save.items():
            if k in ['log_path', 'train_checkpoint_file']:
                config.save[k] = config.save['result_path2'] + '/' + record_name + '/' +  v
            if k in ['valid_sample_path', 'save_weigth_path', 'test_sample_path']:
                config.save[k] = config.save['result_path'] + '/' + record_name + '/' +  v
        self.summary_writer = SummaryWriter(log_dir=config.save['log_path'])
    
    def train_and_valid(self, config):
        if config.local_rank == 0:
            print_log("start train and valid, best_g_loss="+str(self.best_g_loss.item()))
        db_loss = torch.tensor(0.0)
        df_loss = torch.tensor(0.0)
        g_loss = torch.tensor(999999.0)
        total_time = 0.0
        self.trainiter = iter(self.train_loader)
        for step in tqdm(range(config.max_iters)):
            epoch_start_time = time.time()
            if config.parallel_type == 'Distributed' or config.parallel_type == 'Distributed_Apex' or config.parallel_type == 'Horovod':
                # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果
                self.train_loader.sampler.set_epoch(step)
            try:
                data = self.trainiter.next()
            except StopIteration:
                # 避免maxiter比data多
                self.trainiter = iter(self.train_loader)
                data = self.trainiter.next()
            
            db_loss, df_loss = train_D_step(step+1, data, self.G, self.D1, self.D2, self.G_optimizer, self.D1_optimizer, self.D2_optimizer, self.device, config, self.scalar)
            other_loss = None
            if ((step+1) % 2 == 0):
                g_loss, other_loss = train_G_step(step+1, data, self.G, self.D1, self.D2, self.vgg_features, self.G_optimizer, self.D1_optimizer, self.D2_optimizer, self.recognition, self.recognition_criterion, self.device, config, self.scalar)
            train_record = {'G': g_loss.item(), 'D_bg': db_loss.item(), 'D_fu': df_loss.item(), 'best_G': self.best_g_loss.item()}
            if other_loss is not None and config.data_type == 'data_srnet':
                l_t_sk, l_t_l1, l_b_gan, l_b_l1, l_f_gan, l_f_l1, l_f_vgg_per, l_f_vgg_style = other_loss
                train_record.update({
                    'l_t_sk': l_t_sk.item(),
                    'l_t_l1': l_t_l1.item(), 
                    'l_b_gan': l_b_gan.item(), 
                    'l_b_l1': l_b_l1.item(), 
                    'l_f_gan': l_f_gan.item(), 
                    'l_f_l1': l_f_l1.item(), 
                    'l_f_vgg_per': l_f_vgg_per.item(), 
                    'l_f_vgg_style': l_f_vgg_style.item()
                })
            elif other_loss is not None and config.data_type == 'data_imgur5k':
                l_f_gan, l_f_l1, l_f_cyc, l_f_vgg_per, l_f_vgg_style, l_f_tc = other_loss
                train_record.update({
                    'l_f_gan': l_f_gan.item(), 
                    'l_f_l1': l_f_l1.item(), 
                    'l_f_vgg_per': l_f_vgg_per.item(), 
                    'l_f_vgg_style': l_f_vgg_style.item(),
                    'l_f_cyc': l_f_cyc.item(), 
                    'l_f_tc': l_f_tc.item()
                })
                
            if ((step+1) % config.test_interval == 0):
                test_results, metric = test_step_with_label(step+1, self.test_loader, self.G, self.device, config, self.summary_writer)
                mse, psnr, ssim, mse2, psnr2, ssim2, mse3, psnr3, ssim3 = metric
                train_record.update({
                    'mse': mse,
                    'psnr': psnr,
                    'ssim': ssim,
                    'mse2': mse2,
                    'psnr2': psnr2,
                    'ssim2': ssim2,
                    'mse3': mse3,
                    'psnr3': psnr3,
                    'ssim3': ssim3
                })
            if ((step+1) % config.train_log_interval == 0):
                if config.local_rank == 0:
                    for k, v in train_record.items():
                        self.summary_writer.add_scalar('train/'+k, v, step+1)
                    record_epoch(mode='train', epoch=step+1, total_epoch=config.max_iters,
                         record=train_record, record_path=config.save['train_checkpoint_file'])    

            if ((step+1) % config.valid_interval == 0):
                test_results = test_step(step+1, self.valid_loader, self.G, self.device, config, self.summary_writer)
                self.save_result_image(test_results, config.save['valid_sample_path']+'/iter-'+str(step+1).zfill(len(str(config.max_iters))))
            
            epoch_end_time = time.time()
            total_time += epoch_end_time - epoch_start_time
            if config.local_rank == 0:
                print('time cost: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_start_time)), epoch_end_time - epoch_start_time)
                #save
                state = {
                    'step': step+1,
                    'config': config,
                    'best_g_loss': g_loss.item(),
                    'best_db_loss': db_loss.item(),
                    'best_df_loss': df_loss.item(),
                    'generator': get_state_dict(self.G, config),
                    'discriminator1': get_state_dict(self.D1, config),
                    'discriminator2': get_state_dict(self.D2, config),
                    'g_optimizer': get_state_dict(self.G_optimizer, config, isopt=True),
                    'd1_optimizer': get_state_dict(self.D1_optimizer, config, isopt=True),
                    'd2_optimizer': get_state_dict(self.D2_optimizer, config, isopt=True),
                }
                if self.best_g_loss.item() > g_loss.item():
                    print_log("Find Better Model: " + str(g_loss.item()))
                    save_model(step+1, config.max_iters, state, config.model_type,
                           config.save['save_weigth_path'], isBetter=True)
                    self.best_g_loss = g_loss
                if ((step+1) % config.save_ckpt_interval == 0):
                    save_model(step+1, config.max_iters, state, config.model_type,
                           config.save['save_weigth_path'], isBetter=False)



        self.summary_writer.close()
        if config.local_rank == 0:
            print('\ntotal time cost: {}, total_epoch: {}, average time cost: {}'.format(total_time, config.max_iters, total_time/config.max_iters))
 
    def test(self, config):
        #valid_results = test_step(1, self.valid_loader, self.G, self.device, config, self.summary_writer)
        valid_results, metric = test_step_with_label(1, self.valid_loader, self.G, self.device, config, self.summary_writer)
        mse, psnr, ssim, mse2, psnr2, ssim2, mse3, psnr3, ssim3 = metric
        print("valid: mse: {:.6}, psnr: {:.6}, ssim: {:.6} \nmse2: {:.6}, psnr2: {:.6}, ssim2: {:.6} \nmse3: {:.6}, psnr3: {:.6}, ssim3: {:.6}".format(mse, psnr, ssim, mse2, psnr2, ssim2, mse3, psnr3, ssim3))
        self.save_result_image(valid_results, config.save['valid_sample_path']+'/iter-'+str(1).zfill(len(str(1))))
        #test_results = test_step(1, self.test_loader, self.G, self.device, config, self.summary_writer)
        test_results, metric = test_step_with_label(1, self.test_loader, self.G, self.device, config, self.summary_writer)
        mse, psnr, ssim, mse2, psnr2, ssim2, mse3, psnr3, ssim3 = metric
        print("test mse: {:.6}, psnr: {:.6}, ssim: {:.6} \nmse2: {:.6}, psnr2: {:.6}, ssim2: {:.6} \nmse3: {:.6}, psnr3: {:.6}, ssim3: {:.6}".format(mse, psnr, ssim, mse2, psnr2, ssim2, mse3, psnr3, ssim3))
        #self.save_result_image(test_results, config.save['test_sample_path']+'/iter-'+str(1).zfill(len(str(1))))

    def solver_init_seed(self, config):
        print_log('init_seed')
        if config.seed is not None:
            init_seed(config.seed)
        else:
            torch.backends.cudnn.benchmark = True
         
    def solver_init_process_batchsize(self, config): 
         #Distributed_2: 初始化进程组, 设置batchsize
        print_log('init_process_batchsize')
        nprocs = 1
        if config.parallel_type == 'Distributed' or config.parallel_type == 'Distributed_Apex':
            #没有 torch.distributed.launch 读取的默认环境变量作为配置，我们需要手动为 init_process_group 指定参数
            #torch.distributed.init_process_group(backend='nccl')
            torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=config.world_size, rank=config.local_rank)
            nprocs = config.world_size
            config.model_params[config.model_type]['batch_size'] = int(config.model_params[config.model_type]['batch_size']  / nprocs)
            print_log("local_rank: {}, nprocs: {}".format(config.local_rank, nprocs))
        elif config.parallel_type == 'Horovod':
            nprocs = config.world_size
            config.model_params[config.model_type]['batch_size'] = int(config.model_params[config.model_type]['batch_size']  / nprocs)
            print_log("local_rank: {}, nprocs: {}".format(config.local_rank, nprocs))
        return nprocs

    def solver_init_gpu(self, config):
        device = None
        if config.gpu and torch.cuda.is_available():
            if config.parallel_type == 'Distributed' or config.parallel_type == 'Distributed_Apex' or config.parallel_type == 'Horovod':
                device = torch.device('cuda:'+str(config.local_rank))
                torch.cuda.set_device('cuda:'+str(config.local_rank))
            else:
                device = torch.device('cuda:'+str(config.gpu[0]))
                torch.cuda.set_device('cuda:'+str(config.gpu[0]))
        else:
            device = torch.device('cpu')
        print_log('init_gpu: '+str(device))
        return device

    def get_data_loaders(self, config):
        print_log('get_data_loaders')
        dataset_params = config.datasets[config.data_type]
        train_loader = None
        valid_loader = None
        test_loader = None
        if config.mode == 'train':
            train_loader = get_loader(batch_size=config.batch_size,
                                      shuffle=True,
                                      num_workers=config.workers,
                                      mode='train',
                                      data_type = config.data_type,
                                      dataset_params=dataset_params['train'],
                                      parallel_type=config.parallel_type,
                                      data_shape=config.data_shape)
            valid_loader = get_loader(batch_size=config.batch_size,
                                      shuffle=False,
                                      num_workers=config.workers,
                                      mode='valid',
                                      data_type = config.data_type,
                                      dataset_params=dataset_params['valid'],
                                      parallel_type=config.parallel_type,
                                      data_shape=config.data_shape)
            test_loader = get_loader(batch_size=config.batch_size,
                                      shuffle=False,
                                      num_workers=config.workers,
                                      mode='test',
                                      data_type = config.data_type,
                                      dataset_params=dataset_params['test'],
                                      parallel_type=config.parallel_type,
                                      data_shape=config.data_shape)
        elif config.mode == 'test':
            valid_loader = get_loader(batch_size=config.batch_size,
                                      shuffle=False,
                                      num_workers=config.workers,
                                      mode='valid',
                                      data_type = config.data_type,
                                      dataset_params=dataset_params['valid'],
                                      parallel_type=config.parallel_type,
                                      data_shape=config.data_shape)
            test_loader = get_loader(batch_size=config.batch_size,
                                      shuffle=False,
                                      num_workers=config.workers,
                                      mode='test',
                                      data_type = config.data_type,
                                      dataset_params=dataset_params['test'],
                                      parallel_type=config.parallel_type,
                                      data_shape=config.data_shape)
        return train_loader, valid_loader, test_loader

    def solver_get_optimizer(self, config):
        print_log('get_optimizer')
        if config.parallel_type == 'Horovod':
            self.G_optimizer = get_optimizer(self.G, config.optimizer_type, config.optimizer_params[config.optimizer_type])
            self.D1_optimizer = get_optimizer(self.D1, config.optimizer_type, config.optimizer_params[config.optimizer_type])
            self.D2_optimizer = get_optimizer(self.D2, config.optimizer_type, config.optimizer_params[config.optimizer_type])
            hvd.broadcast_optimizer_state(self.G_optimizer, root_rank=0) #？？？
            hvd.broadcast_optimizer_state(self.D1_optimizer, root_rank=0)
            hvd.broadcast_optimizer_state(self.D2_optimizer, root_rank=0)
            compression = hvd.Compression.fp16
            self.G_optimizer = hvd.DistributedOptimizer(self.G_optimizer, named_parameters=self.G.named_parameters(), compression=compression)
            self.D1_optimizer = hvd.DistributedOptimizer(self.D1_optimizer, named_parameters=self.D1.named_parameters(), compression=compression)
            self.D2_optimizer = hvd.DistributedOptimizer(self.D2_optimizer, named_parameters=self.D2.named_parameters(), compression=compression)
        else:
            self.G_optimizer = get_optimizer(self.G, config.optimizer_type, config.optimizer_params[config.optimizer_type])
            self.D1_optimizer = get_optimizer(self.D1, config.optimizer_type, config.optimizer_params[config.optimizer_type])
            self.D2_optimizer = get_optimizer(self.D2, config.optimizer_type, config.optimizer_params[config.optimizer_type])
        return self.G_optimizer, self.D1_optimizer, self.D2_optimizer   

    def load_weights(self, config):
        print_log("load weights: {}".format(config.load['load_weight_path']))
        checkpoint = torch.load(config.load['load_weight_path'], map_location=self.device)
        state_dict = checkpoint['generator']
        pretrained_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}	
        self.G.load_state_dict(pretrained_dict)
        state_dict = checkpoint['discriminator1']
        pretrained_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}	
        self.D1.load_state_dict(pretrained_dict)
        state_dict = checkpoint['discriminator2']
        pretrained_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}	
        self.D2.load_state_dict(pretrained_dict)
        state_dict = checkpoint['g_optimizer']
        pretrained_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}	
        self.G_optimizer.load_state_dict(pretrained_dict)
        state_dict = checkpoint['d1_optimizer']
        pretrained_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}	
        self.D1_optimizer.load_state_dict(pretrained_dict)
        state_dict = checkpoint['d2_optimizer']
        pretrained_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}	
        self.D2_optimizer.load_state_dict(pretrained_dict)
        if checkpoint.get('best_g_loss') is not None:
            print_log("load best G loss: {}".format(checkpoint['best_g_loss']))
            self.best_g_loss = torch.tensor(checkpoint['best_g_loss'])

    def save_result_image(self, results, save_path):
        for result in results:
            i_t, i_s, o_sk, o_t, o_b, o_f, t_f, img_name = result
            for i in range(o_sk.shape[0]):
                i_t_single = F.to_pil_image((i_t[i] + 1)/2)
                i_s_single = F.to_pil_image((i_s[i] + 1)/2)
                o_sk_single = F.to_pil_image(o_sk[i])
                o_t_single = F.to_pil_image((o_t[i] + 1)/2)
                o_b_single = F.to_pil_image((o_b[i] + 1)/2)
                o_f_single = F.to_pil_image((o_f[i] + 1)/2)
                img_name_single = '.'.join(img_name[i].split('.')[:-1])
                check_dir(os.path.join(save_path, img_name_single))
                i_t_single.save(os.path.join(save_path, img_name_single, 'i_t.png'))        
                i_s_single.save(os.path.join(save_path, img_name_single, 'i_s.png'))       
                o_sk_single.save(os.path.join(save_path, img_name_single, 'o_sk.png'))
                o_t_single.save(os.path.join(save_path, img_name_single, 'o_t.png'))
                o_b_single.save(os.path.join(save_path, img_name_single, 'o_b.png'))
                o_f_single.save(os.path.join(save_path, img_name_single, 'o_f.png')) 
                if t_f is not None:
                    t_f_single = F.to_pil_image((t_f[i] + 1)/2)
                    t_f_single.save(os.path.join(save_path, img_name_single, 't_f.png'))

    def init_parallel(self, config):
        # model paralle
        if torch.cuda.device_count() > 1:
            print_log("use parallel type: {}".format(config.parallel_type))
            if config.parallel_type == 'DataParallel':
                self.G = torch.nn.DataParallel(
                    self.G, device_ids=config.gpu, output_device=config.gpu[0])
                self.D1 = torch.nn.DataParallel(
                    self.D1, device_ids=config.gpu, output_device=config.gpu[0])
                self.D2 = torch.nn.DataParallel(
                    self.D2, device_ids=config.gpu, output_device=config.gpu[0])
                #self.G_optimizer = torch.nn.DataParallel(
                #    self.G_optimizer, device_ids=config.gpu, output_device=config.gpu[0])
                #self.D1_optimizer = torch.nn.DataParallel(
                #    self.D1_optimizer, device_ids=config.gpu, output_device=config.gpu[0])
                #self.D2_optimizer = torch.nn.DataParallel(
                #    self.D2_optimizer, device_ids=config.gpu, output_device=config.gpu[0])  
