import torch
import random
import numpy as np

from pathlib import Path
from tqdm import tqdm
from torch import nn

from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from lion_pytorch import Lion


from vae_mda.logger.vae_train_logger import VAETrainCML
from vae_mda.utils.loss import (
    MultiscaleBlockSiconcLoss,
    MultiscaleSiconcAuxLoss,
    MultiscaleSiconcLoss,
    SiconcAuxLoss,
    SimpleLoss,
)

losses = {
    loss.name: loss
    for loss in (
        SimpleLoss,
        SiconcAuxLoss,
        MultiscaleSiconcLoss,
        MultiscaleBlockSiconcLoss,
        MultiscaleSiconcAuxLoss,
    )
}
        
class Trainer():
    @staticmethod
    def _move_model_to_device(model, device):
        model = model.to(device)
        if hasattr(model, 'model'):
            model.model = model.model.to(device)
        return model

    @staticmethod
    def _resolve_module(root, module_path):
        module = root
        for attr in module_path.split('.'):
            if not hasattr(module, attr):
                raise ValueError(f"Unknown module path to freeze: {module_path}")
            module = getattr(module, attr)
        return module

    def init_loss(self, config):
        # todo вынести в отдельный класс, пересчитать на 1 число
        loss_class = losses[config['loss_type']]
        param = loss_class.init_params(config, self.transformer)
        self.loss = loss_class(**param).to(self.device)

    def freeze_modules(self, config):
        freeze_modules = config.get('freeze_modules', [])
        if not freeze_modules:
            return
        for module_path in freeze_modules:
            module = self._resolve_module(self.model, module_path)
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    @staticmethod
    def _base_dataset(dataset):
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        return dataset

    def init_data(self, train_ds, val_ds, b_size):
        self.train_dl = DataLoader(
            train_ds, batch_size=b_size, shuffle=True, num_workers=8,
        )
        self.val_dl = DataLoader(
            val_ds, batch_size=b_size, shuffle=False, num_workers=8,
        )
        self.transformer = self._base_dataset(train_ds).transformer
        self.mask = self.transformer.mask

    def init_optimizer(self, config):
        self.lr = config['lr']
        lr_warmup_steps = 4*len(self.train_dl)
        self.num_epochs = config['epoch']
        num_training_steps = (len(self.train_dl) * self.num_epochs)
        trainable_params = [
            param for param in self.model.parameters() if param.requires_grad
        ]
        if not trainable_params:
            raise ValueError('No trainable parameters left after freezing modules')

        if config['optimizer'] == 'adam':
            self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)
        elif config['optimizer'] == 'lion':
            self.optimizer = Lion(trainable_params, lr=self.lr)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def __init__(self, config, model, ds_train, ds_val, data_config, model_config):
        print("Run Trainer init")
        self.logger = VAETrainCML(config, model_config, data_config)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = self._move_model_to_device(model, self.device)
        self.train_use_mode = config.get('train_use_mode', False)
        self.latent_noise_std = config.get('latent_noise_std', 0.0)
        # data
        self.init_data(ds_train, ds_val, config['batch_size'])
        if 'add_gauss' not in config:
            config['add_gauss'] = False
        if config['add_gauss']:
            ds_train.transformer.add_gauss = True
        self.freeze_modules(config)
        # loss
        self.init_loss(config)
        # optimizer
        self.init_optimizer(config)
        self.log_dir = Path(self.model.path)

    def forward(self, batch, train=False):
        x, mask, t = batch
        torch.autograd.set_detect_anomaly(True)
        x_true = x.detach().to(self.device)
        x = x.to(self.device)
        t = t.to(self.device)
        if train and not self.train_use_mode:
            lat, kl_loss = self.model.encode_sample(x)
        else: 
            lat, kl_loss = self.model.encode(x)
        if train and self.latent_noise_std > 0.0:
            lat = lat + self.latent_noise_std * torch.randn_like(lat)
        if len(lat.shape) == 4:
            norm = lat.shape[0]*lat.shape[1]*lat.shape[2]*lat.shape[3]
        elif len(lat.shape) == 2:
            norm = lat.shape[0]*lat.shape[1]
        kl_loss /= norm
        rec = self.model.decode(lat, t)
        aux_outputs = {}
        if hasattr(self.model, 'get_aux_outputs'):
            aux_outputs = self.model.get_aux_outputs() or {}
        loss, logs = self.loss(rec, x_true, mask, kl_loss, **aux_outputs)
        self.logger.log_loss(logs, train)
        logs['lr'] = self.lr_scheduler.get_last_lr()[0]

        return x_true, rec, loss, logs

    def train_step(self, batch):
        x, rec, loss, logs = self.forward(batch, train=True)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return logs

    def eval_step(self, batch):
        _, mask, t = batch
        x, rec, loss, logs = self.forward(batch, train=False)
        x = self.transformer.detransform(x.detach().cpu(), mask.cpu())
        rec = self.transformer.detransform(rec.detach().cpu(), mask.cpu())
        self.logger.validate(rec, x)

    def show_random(self):
        inx = random.randint(0, len(self.val_dl.dataset)-1)
        batch = self.val_dl.dataset[inx]
        x, mask, t = batch
        x, mask, t = x[None, ...], mask[None, ...], torch.tensor([t])
        batch = x, mask, t
        x, rec, loss, logs = self.forward(batch, train=False)
        x = self.transformer.detransform(x.detach().cpu(), mask)
        rec = self.transformer.detransform(rec.detach().cpu(), mask)
        self.logger.log_img(rec[0], x[0])
        torch.cuda.empty_cache()

    def train_epoch(self, epoch):
        self.model = self.model.to(self.device)
        self.model.train()
        progress_bar = tqdm(
            total=len(self.train_dl),
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for batch in self.train_dl:
            logs = self.train_step(batch)
            logs["global_step"] = self.global_step
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            self.global_step += 1
                
        batch = None
        torch.cuda.empty_cache()

    def eval_epoch(self, epoch):
        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_dl:
                self.eval_step(batch)

        batch = None
        torch.cuda.empty_cache()
        self.show_random() 
        return self.logger.get_val_loss()

    def test_epoch(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_dl:
                self.eval_step(batch)

        self.logger.end_test()
        batch = None
        torch.cuda.empty_cache()

    def train(self):
        self.global_step = 0
        beast_val_loss = 1_000_000_000_000_000
        steps_without_improvement = 0
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)
            val_loss = self.eval_epoch(epoch)
            self.logger.end_epoch()
            if val_loss<beast_val_loss:
                print(f'Save model: {self.log_dir}')
                self.model.save_pretrained(self.log_dir)
                beast_val_loss = val_loss
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
            if steps_without_improvement >=30:
                print('NN overfitting')
                break
        self.model.from_pretrained(self.log_dir)
        self.model = self._move_model_to_device(self.model, self.device)
        self.test_epoch()
        self.logger.close()
