import argparse
from configs.config import load_config
import torch
# General config

# import models.OccNet as OccNet
# import models.NormNet as NormNet
# import models.ColorNet as ColorNet

from models.OccNet import OccNet
from models.NormNet import NormNet
from models.ColorNet import ColorNet
from models.RBF import RBF
from data.ARCHData import ARCHData
from data.MGNData import MGNData

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

def train(opt):

    # train_dataset = ARCHData('train', data_path=opt['data_path'], split_file=opt['split_file'], batch_size=opt['training']['batch_size'], num_workers=opt['training']['num_worker'])
    # val_dataset = ARCHData('val', data_path=opt['data_path'], split_file=opt['split_file'], batch_size=opt['training']['batch_size'], num_workers=opt['training']['num_worker'])

    train_dataset = MGNData('train', data_path=opt['data']['data_dir'], split_file=opt['data']['split_file'], batch_size=opt['training']['batch_size'], num_workers=opt['training']['num_worker'], split=True, num_view=360)
    val_dataset = MGNData('val', data_path=opt['data']['data_dir'], split_file=opt['data']['split_file'], batch_size=opt['training']['batch_size'], num_workers=opt['training']['num_worker'], split=False, num_view=360)

    print("Datasets -> Created.")

    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device -> {device}")

    from models.train_arch import Trainer as arch_trainer
    trainer = arch_trainer(occ_net=OccNet(opt['model']['occ_net'], device), norm_net=NormNet(opt['model']['normal_net'], device), col_net=ColorNet(opt['model']['color_net'], device), rbf=RBF, device=device, train_dataset=train_dataset, val_dataset=val_dataset, batch_size=opt['training']['batch_size'], opt=opt)
    print("Trainer -> Created.")

    trainer.train_model(opt['training']['max_epoch'], eval=opt['training']['eval'])

def trainWrapper(args=None):
    parser = argparse.ArgumentParser(
        description='Train ARCH.'
    )
    parser.add_argument('--config', '-c', type=str, help='Path to config file.')
    args = parser.parse_args()

    opt = load_config(args.config, './configs/arch.yaml')

    train(opt)

if __name__ == '__main__':
    trainWrapper()
