from typing import *
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from torch.nn.utils import clip_grad_norm_


class NaiveModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.SELU(),
        )
    
    def forward(self, x):
        return self.layers(x)

class MyTabNet(pl.LightningModule):

    lambda_sparse: float = 1e-3
    loss_fn = nn.MSELoss()

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = self.hparams.config.task
        # self.automatic_optimization = False # activates manual optimization!
        # for manual scheduler stepping forward
        # self._last_sch_mon_val: torch.Tensor = None

        if hasattr(self.cfg, 'optimizer'):
            self.clip_value = self.cfg.optimizer.grad_clip_value

        model_cfg = self.cfg.model
        self.net = NaiveModel(model_cfg.input_dim, model_cfg.output_dim, model_cfg.hidden_dim)

        # freeze
        if hasattr(self.cfg, 'freeze'):
            for m in self.cfg.freeze:
                module = getattr(self, m)
                module.freeze()
        
    def forward(self, x_batch):
        return self.net(x_batch)

    def _common_step(self, batch) -> Dict:
        x, y_gt = batch['x'], batch['y_label']
        y_pred = self.forward(x)
        
        out = {}
        out['loss'] = self._compute_loss(y_pred, y_gt)
        out['y_pred'] = y_pred.detach()
        
        return out
    
    def _compute_loss(self, y_pred, y_gt):
        return self.loss_fn(y_pred, y_gt)

    def _compute_and_log_metrics(self, y_pred, y_gt, mode: str):
        y_pred = torch.flatten(y_pred)
        y_gt = torch.flatten(y_gt)
        # prec
        prec = torch.sum(torch.abs(y_gt - y_pred) < 1.1e-2) / y_pred.shape[0]
        self.log(f'{mode}_prec', prec, on_step=True)
        # MAE
        mae_fn = nn.L1Loss()
        mae = mae_fn(y_pred, y_gt)
        self.log(f'{mode}_MAE', mae, on_step=True)

        return prec, mae

    def _manual_optimize(self, loss: torch.Tensor):
        opt = self.optimizers()
        opt.zero_grad()
        # manual backward
        loss.backward()
        if self.clip_value > 0:
            clip_grad_norm_(self.net.parameters(), self.clip_value)
        # update
        opt.step()
        if self._last_sch_mon_val is not None:
            sch = self.lr_schedulers()
            if isinstance(sch, optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(self._last_sch_mon_val)
            else:
                sch.step()

    def training_step(self, batch, batch_idx) -> Dict:
        out = self._common_step(batch)
        self._manual_optimize(out['loss'])
        # log loss
        self.log('train_loss', out['loss'], prog_bar=True, on_step=True)
        self.log('train_loss_M', out['M_loss'], prog_bar=True, on_step=True)
        # log lr of optimizer
        opt = self.optimizers()
        self.log(
            f'lr', opt.param_groups[0]['lr'],
            prog_bar=True, on_step=True,
        )

        self._compute_and_log_metrics(out['y_pred'], batch[1], 'train')

        return out

    def validation_step(self, batch, batch_idx) -> Dict:
        out = self._common_step(batch)
        self.log('val_loss', out['loss'], prog_bar=True, on_step=True)
        self.log('val_loss_M', out['M_loss'], prog_bar=True, on_step=True)

        prec, mae = self._compute_and_log_metrics(out['y_pred'], batch[1], 'val')
        self._last_sch_mon_val = mae

        return out

    def test_step(self, batch, batch_idx):
        out = self._common_step(batch)
        self.log('test_loss', out['loss'], prog_bar=True, on_step=True)
        self.log('test_loss_M', out['M_loss'], prog_bar=True, on_step=True)

        prec, mae = self._compute_and_log_metrics(out['y_pred'], batch[1], 'test')
        self.test_metrics['prec'].append(prec)
        self.test_metrics['mae'].append(mae)
        self.original_data.append(out['data'])

        return out

    def on_test_epoch_start(self) -> None:
        self.test_metrics = {'prec': [], 'mae': []}
        self.original_data = []
        return super().on_test_epoch_start()

    def test_epoch_end(self, outputs):
        from IPython import embed
        embed()
        prec = self.test_metrics['prec']
        mae = self.test_metrics['mae']
        prec = torch.mean(torch.stack(prec).flatten())
        mae = torch.mean(torch.stack(mae).flatten())
        self.log('test_res_prec', prec)
        self.log('test_res_mae', mae)
        self._output_res()
        return super().test_epoch_end(outputs)

    def _output_res():
        pass

    def configure_optimizers(self):
        optim_cfg = self.cfg.optimizer
        # net params to train
        params_list = list(self.net.parameters())

        optim_params = []
        optim_params.append({
            'params': params_list,
            'lr': optim_cfg.lr,
            'betas': optim_cfg.betas,
            'eps': optim_cfg.eps,
            'weight_decay': optim_cfg.weight_decay,
        })
        optimizer = optim.AdamW(optim_params)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=optim_cfg.patience,
            factor=optim_cfg.factor,
            threshold=optim_cfg.threshold,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': optim_cfg.monitor_val,
            },
        }
    