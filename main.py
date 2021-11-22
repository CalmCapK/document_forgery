import argparse
import horovod.torch as hvd
import torch.multiprocessing as mp
import yaml

from pipelines.solver import Solver
from tools.utils import print_log

#@profile
def main(local_rank, config):
    print_log("local_rank:" + str(local_rank))
    config.local_rank = local_rank
    solver = Solver(config)
    if config.mode == 'train':
        solver.train_and_valid(config)
    else:
        solver.test(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='./configs/config.yaml')
    #Distributed_1: local_rank代表当前进程，分布式启动会会自动更新该参数,不需要在命令行显式调用
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)
    
    #single, DataParallel, Distributed, Distributed_Apex
    main(args.local_rank, argparse.Namespace(**config))