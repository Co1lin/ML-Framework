import os
import random
import hydra
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
import torch
from model.MyTabNet import MyTabNet
from data.MyDM import MyDM

def init_wandb(cfg):
    # os.environ['WANDB_API_KEY'] = cfg.wandb.key
    wandb_logger = WandbLogger(
        # entity=cfg.wandb.entity,
        offline=cfg.wandb.offline,
        project=cfg.wandb.proj,
        group=cfg.task.name,
    )
    return wandb_logger

model: pl.LightningModule = None
datamodule: pl.LightningDataModule = None
ckpt_callback_list: list = None
output_dir: str = None

def train(cfg):
    wandb_logger = init_wandb(cfg)
    ckpt_callback_list = [
        ModelCheckpoint(
            monitor=cfg.task.optimizer.monitor_val,
            dirpath=output_dir,
            filename=f'{cfg.task.name}-' + '{epoch}-{val_metric:.2f}-best',
            save_top_k=3,
            save_last=True,
            mode='min',
        ),
    ]
    trainer = pl.Trainer(
        resume_from_checkpoint=cfg.weight if cfg.task.resume else None,
        max_epochs=cfg.task.epochs,
        gpus=cfg.gpus,
        logger=wandb_logger, 
        strategy='ddp', 
        log_every_n_steps=10, 
        callbacks=ckpt_callback_list,
        sync_batchnorm=True,
        grad_clip_value=cfg.task.optimizer.grad_clip_value,
        # val_check_interval=cfg.val_check_interval,
        # plugins=DDPPlugin(find_unused_parameters=False),
    )
    trainer.fit(model, datamodule=datamodule)

def test(cfg):
    wandb_logger = init_wandb(cfg)
    trainer = pl.Trainer(
        gpus=cfg.gpus[:1],
        logger=wandb_logger,
    )
    trainer.test(model, datamodule=datamodule)


def seed_all(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    # increase ulimit
    # import resource
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (16384, rlimit[1]))

    global model
    global datamodule
    global ckpt_callback_list
    global output_dir

    output_dir = os.path.abspath(os.curdir) # get hydra output dir
    os.chdir(hydra.utils.get_original_cwd())    # set working dir to the original one

    seed_all(cfg.seed)
    
    if cfg.task.mode == 'train':
        if cfg.task.resume:
            model = MyTabNet.load_from_checkpoint(cfg.weight)
        else:
            if not cfg.task.finetune or not os.path.exists(cfg.weight):
                model = MyTabNet(cfg)  # train from scratch
            else:   # finetuning based on previous weights
                # set strict to False to ignore the missing params in the previous phase
                model = MyTabNet.load_from_checkpoint(cfg.weight, config=cfg, strict=False)
    elif cfg.task.mode == 'test':
        if os.path.exists(cfg.weight):
            model = MyTabNet.load_from_checkpoint(cfg.weight, config=cfg)
        else:
            model = MyTabNet(cfg)  # test from scratch
    else:
        raise ValueError(f'Invalid task mode: {cfg.task.mode}')
    
    datamodule = MyDM(cfg)

    mode = cfg.task.mode
    if mode == 'train':
        train(cfg)
    elif mode == 'test':
        test(cfg)


if __name__ == '__main__':
    main()

# python main.py task=train gpus='[0,1]' 'weight=""'