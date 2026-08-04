import torch
import torch.nn.functional as F

from torch import nn 
from abc import ABC, abstractmethod, abstractstaticmethod


class VAELoss(ABC):
    name = ''

    @abstractstaticmethod
    def init_params(config, mask):
        pass

    @abstractmethod
    def forward(self, x_rec, x_true, mask, kl_loss, **kwargs):
        pass


class SimpleLoss(VAELoss, nn.Module):
    name = 'simple'
    def __init__(
        self, masked_loss, kl_weight
    ):
        super().__init__()
        self.recon_loss = nn.MSELoss(reduction='mean')
        self.masked_loss = masked_loss
        self.kl_weight = kl_weight
    
    def forward(self, x_rec, x_true, mask, kl_loss, **kwargs):
        loss = 0
        
        if self.masked_loss:
            recon_loss = self.recon_loss(
                x_rec[~mask],
                x_true[~mask],
            )
        else: 
            recon_loss = self.recon_loss(
                x_rec, x_true
            )
        
        loss += recon_loss + self.kl_weight*kl_loss
        
        logs =  {
            'loss': loss.detach().item(),
            'recon_loss': recon_loss.detach().item(),
            'kl_loss': kl_loss.detach().item(),
        }
        return loss, logs
    
    @staticmethod
    def init_params(config, transformer):
        return {
            'masked_loss': config['masked_loss'],
            'kl_weight': config['kl_weight'],
        }


class SiconcAuxLoss(VAELoss, nn.Module):
    name = 'siconc_aux'

    def __init__(
        self,
        masked_loss,
        kl_weight,
        siconc_channel,
        siconc_mean,
        siconc_std,
        sobel_weight=0.0,
        binary_weight=0.0,
    ):
        super().__init__()
        self.recon_loss = nn.MSELoss(reduction='mean')
        self.sobel_metric = nn.L1Loss(reduction='mean')
        self.masked_loss = masked_loss
        self.kl_weight = kl_weight
        self.siconc_channel = siconc_channel
        self.siconc_mean = float(siconc_mean)
        self.siconc_std = float(siconc_std)
        self.sobel_weight = sobel_weight
        self.binary_weight = binary_weight

        sobel_x = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _denormalize_siconc(self, x):
        x = x * self.siconc_std + self.siconc_mean
        return torch.clamp(x, 0.0, 1.0)

    def _sobel(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)

    def forward(self, x_rec, x_true, mask, kl_loss, **kwargs):
        if self.masked_loss:
            recon_loss = self.recon_loss(
                x_rec[~mask],
                x_true[~mask],
            )
        else:
            recon_loss = self.recon_loss(x_rec, x_true)

        pred_siconc = self._denormalize_siconc(
            x_rec[:, self.siconc_channel:self.siconc_channel + 1]
        )
        true_siconc = self._denormalize_siconc(
            x_true[:, self.siconc_channel:self.siconc_channel + 1]
        )

        channel_mask = mask[:, self.siconc_channel:self.siconc_channel + 1]
        sobel_loss = kl_loss.new_tensor(0.0)
        binary_loss = kl_loss.new_tensor(0.0)

        if self.sobel_weight > 0.0:
            pred_edges = self._sobel(pred_siconc)
            true_edges = self._sobel(true_siconc)
            if self.masked_loss:
                sobel_loss = self.sobel_metric(
                    pred_edges[~channel_mask],
                    true_edges[~channel_mask],
                )
            else:
                sobel_loss = self.sobel_metric(pred_edges, true_edges)

        if self.binary_weight > 0.0:
            if self.masked_loss:
                binary_loss = (pred_siconc * (1.0 - pred_siconc))[~channel_mask].mean()
            else:
                binary_loss = (pred_siconc * (1.0 - pred_siconc)).mean()

        loss = (
            recon_loss
            + self.kl_weight * kl_loss
            + self.sobel_weight * sobel_loss
            + self.binary_weight * binary_loss
        )

        logs = {
            'loss': loss.detach().item(),
            'recon_loss': recon_loss.detach().item(),
            'kl_loss': kl_loss.detach().item(),
            'sobel_loss': sobel_loss.detach().item(),
            'binary_loss': binary_loss.detach().item(),
        }
        return loss, logs

    @staticmethod
    def init_params(config, transformer):
        channel = config.get('siconc_channel', 0)
        mean = config.get('siconc_mean')
        std = config.get('siconc_std')
        if mean is None:
            mean = transformer.means[channel].item()
        if std is None:
            std = transformer.stds[channel].item()
        return {
            'masked_loss': config['masked_loss'],
            'kl_weight': config['kl_weight'],
            'siconc_channel': channel,
            'siconc_mean': mean,
            'siconc_std': std,
            'sobel_weight': config.get('sobel_weight', 0.0),
            'binary_weight': config.get('binary_weight', 0.0),
        }


class MultiscaleSiconcLoss(VAELoss, nn.Module):
    name = 'multiscale_siconc'

    def __init__(
        self,
        masked_loss,
        kl_weight,
        siconc_channel,
        siconc_mean,
        siconc_std,
        ms_weight=0.2,
        ms_scales=None,
        ms_scale_weights=None,
    ):
        super().__init__()
        self.recon_loss = nn.MSELoss(reduction='mean')
        self.ms_metric = nn.MSELoss(reduction='mean')
        self.masked_loss = masked_loss
        self.kl_weight = kl_weight
        self.siconc_channel = siconc_channel
        self.siconc_mean = float(siconc_mean)
        self.siconc_std = float(siconc_std)
        self.ms_weight = ms_weight
        self.ms_scales = ms_scales if ms_scales is not None else [1, 2, 4]
        self.ms_scale_weights = (
            ms_scale_weights if ms_scale_weights is not None else [1.0, 0.5, 0.25]
        )
        if len(self.ms_scales) != len(self.ms_scale_weights):
            raise ValueError('ms_scales and ms_scale_weights must have same length')

    def _denormalize_siconc(self, x):
        x = x * self.siconc_std + self.siconc_mean
        return torch.clamp(x, 0.0, 1.0)

    def _pool(self, x, scale):
        if scale == 1:
            return x
        return F.avg_pool2d(x, kernel_size=scale, stride=scale)

    def _multiscale_loss(self, pred, true):
        loss = pred.new_tensor(0.0)
        for scale, weight in zip(self.ms_scales, self.ms_scale_weights):
            pred_scaled = self._pool(pred, scale)
            true_scaled = self._pool(true, scale)
            loss = loss + weight * self.ms_metric(pred_scaled, true_scaled)
        return loss

    def forward(self, x_rec, x_true, mask, kl_loss, **kwargs):
        if self.masked_loss:
            recon_loss = self.recon_loss(
                x_rec[~mask],
                x_true[~mask],
            )
        else:
            recon_loss = self.recon_loss(x_rec, x_true)

        pred_siconc = self._denormalize_siconc(
            x_rec[:, self.siconc_channel:self.siconc_channel + 1]
        )
        true_siconc = self._denormalize_siconc(
            x_true[:, self.siconc_channel:self.siconc_channel + 1]
        )
        ms_loss = self._multiscale_loss(pred_siconc, true_siconc)

        loss = recon_loss + self.kl_weight * kl_loss + self.ms_weight * ms_loss
        logs = {
            'loss': loss.detach().item(),
            'recon_loss': recon_loss.detach().item(),
            'kl_loss': kl_loss.detach().item(),
            'ms_loss': ms_loss.detach().item(),
        }
        return loss, logs

    @staticmethod
    def init_params(config, transformer):
        channel = config.get('siconc_channel', 0)
        mean = config.get('siconc_mean')
        std = config.get('siconc_std')
        if mean is None:
            mean = transformer.means[channel].item()
        if std is None:
            std = transformer.stds[channel].item()
        return {
            'masked_loss': config['masked_loss'],
            'kl_weight': config['kl_weight'],
            'siconc_channel': channel,
            'siconc_mean': mean,
            'siconc_std': std,
            'ms_weight': config.get('ms_weight', 0.2),
            'ms_scales': config.get('ms_scales', [1, 2, 4]),
            'ms_scale_weights': config.get('ms_scale_weights', [1.0, 0.5, 0.25]),
        }


class MultiscaleBlockSiconcLoss(VAELoss, nn.Module):
    name = 'multiscale_block_siconc'

    def __init__(
        self,
        masked_loss,
        kl_weight,
        siconc_channel,
        siconc_mean,
        siconc_std,
        ms_weight=0.2,
        ms_scales=None,
        ms_scale_weights=None,
    ):
        super().__init__()
        self.recon_loss = nn.MSELoss(reduction='mean')
        self.fine_metric = nn.L1Loss(reduction='mean')
        self.coarse_metric = nn.L1Loss(reduction='mean')
        self.masked_loss = masked_loss
        self.kl_weight = kl_weight
        self.siconc_channel = siconc_channel
        self.siconc_mean = float(siconc_mean)
        self.siconc_std = float(siconc_std)
        self.ms_weight = ms_weight
        self.ms_scales = ms_scales if ms_scales is not None else [1, 2, 4]
        self.ms_scale_weights = (
            ms_scale_weights if ms_scale_weights is not None else [1.0, 1.0, 0.5]
        )
        if len(self.ms_scales) != len(self.ms_scale_weights):
            raise ValueError('ms_scales and ms_scale_weights must have same length')

    def _denormalize_siconc(self, x):
        x = x * self.siconc_std + self.siconc_mean
        return torch.clamp(x, 0.0, 1.0)

    @staticmethod
    def _pool(x, scale):
        if scale == 1:
            return x
        return F.avg_pool2d(x, kernel_size=scale, stride=scale)

    def _multiscale_loss(self, pred, true):
        loss = pred.new_tensor(0.0)
        for scale, weight in zip(self.ms_scales, self.ms_scale_weights):
            if scale == 1:
                scale_loss = self.fine_metric(pred, true)
            else:
                pred_scaled = self._pool(pred, scale)
                true_scaled = self._pool(true, scale)
                scale_loss = self.coarse_metric(pred_scaled, true_scaled)
            loss = loss + weight * scale_loss
        return loss

    def forward(self, x_rec, x_true, mask, kl_loss, **kwargs):
        if self.masked_loss:
            recon_loss = self.recon_loss(
                x_rec[~mask],
                x_true[~mask],
            )
        else:
            recon_loss = self.recon_loss(x_rec, x_true)

        pred_siconc = self._denormalize_siconc(
            x_rec[:, self.siconc_channel:self.siconc_channel + 1]
        )
        true_siconc = self._denormalize_siconc(
            x_true[:, self.siconc_channel:self.siconc_channel + 1]
        )
        ms_loss = self._multiscale_loss(pred_siconc, true_siconc)

        loss = recon_loss + self.kl_weight * kl_loss + self.ms_weight * ms_loss
        logs = {
            'loss': loss.detach().item(),
            'recon_loss': recon_loss.detach().item(),
            'kl_loss': kl_loss.detach().item(),
            'ms_block_loss': ms_loss.detach().item(),
        }
        return loss, logs

    @staticmethod
    def init_params(config, transformer):
        channel = config.get('siconc_channel', 0)
        mean = config.get('siconc_mean')
        std = config.get('siconc_std')
        if mean is None:
            mean = transformer.means[channel].item()
        if std is None:
            std = transformer.stds[channel].item()
        return {
            'masked_loss': config['masked_loss'],
            'kl_weight': config['kl_weight'],
            'siconc_channel': channel,
            'siconc_mean': mean,
            'siconc_std': std,
            'ms_weight': config.get('ms_weight', 0.2),
            'ms_scales': config.get('ms_scales', [1, 2, 4]),
            'ms_scale_weights': config.get('ms_scale_weights', [1.0, 1.0, 0.5]),
        }


class MultiscaleSiconcAuxLoss(VAELoss, nn.Module):
    name = 'multiscale_siconc_aux'

    def __init__(
        self,
        masked_loss,
        kl_weight,
        siconc_channel,
        siconc_mean,
        siconc_std,
        ms_weight=0.1,
        ms_scales=None,
        ms_scale_weights=None,
        sobel_weight=0.4,
        binary_weight=0.4,
    ):
        super().__init__()
        self.recon_loss = nn.MSELoss(reduction='mean')
        self.ms_metric = nn.MSELoss(reduction='mean')
        self.sobel_metric = nn.L1Loss(reduction='mean')
        self.masked_loss = masked_loss
        self.kl_weight = kl_weight
        self.siconc_channel = siconc_channel
        self.siconc_mean = float(siconc_mean)
        self.siconc_std = float(siconc_std)
        self.ms_weight = ms_weight
        self.ms_scales = ms_scales if ms_scales is not None else [1, 2, 4]
        self.ms_scale_weights = (
            ms_scale_weights if ms_scale_weights is not None else [1.0, 0.5, 0.25]
        )
        self.sobel_weight = sobel_weight
        self.binary_weight = binary_weight

        sobel_x = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _denormalize_siconc(self, x):
        x = x * self.siconc_std + self.siconc_mean
        return torch.clamp(x, 0.0, 1.0)

    @staticmethod
    def _pool(x, scale):
        if scale == 1:
            return x
        return F.avg_pool2d(x, kernel_size=scale, stride=scale)

    def _multiscale_loss(self, pred, true):
        loss = pred.new_tensor(0.0)
        for scale, weight in zip(self.ms_scales, self.ms_scale_weights):
            pred_scaled = self._pool(pred, scale)
            true_scaled = self._pool(true, scale)
            loss = loss + weight * self.ms_metric(pred_scaled, true_scaled)
        return loss

    def _sobel(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)

    def forward(self, x_rec, x_true, mask, kl_loss, **kwargs):
        if self.masked_loss:
            recon_loss = self.recon_loss(x_rec[~mask], x_true[~mask])
        else:
            recon_loss = self.recon_loss(x_rec, x_true)

        pred_siconc = self._denormalize_siconc(
            x_rec[:, self.siconc_channel:self.siconc_channel + 1]
        )
        true_siconc = self._denormalize_siconc(
            x_true[:, self.siconc_channel:self.siconc_channel + 1]
        )
        channel_mask = mask[:, self.siconc_channel:self.siconc_channel + 1]

        ms_loss = self._multiscale_loss(pred_siconc, true_siconc)
        pred_edges = self._sobel(pred_siconc)
        true_edges = self._sobel(true_siconc)
        if self.masked_loss:
            sobel_loss = self.sobel_metric(pred_edges[~channel_mask], true_edges[~channel_mask])
            binary_loss = (pred_siconc * (1.0 - pred_siconc))[~channel_mask].mean()
        else:
            sobel_loss = self.sobel_metric(pred_edges, true_edges)
            binary_loss = (pred_siconc * (1.0 - pred_siconc)).mean()

        loss = (
            recon_loss
            + self.kl_weight * kl_loss
            + self.ms_weight * ms_loss
            + self.sobel_weight * sobel_loss
            + self.binary_weight * binary_loss
        )
        logs = {
            'loss': loss.detach().item(),
            'recon_loss': recon_loss.detach().item(),
            'kl_loss': kl_loss.detach().item(),
            'ms_loss': ms_loss.detach().item(),
            'sobel_loss': sobel_loss.detach().item(),
            'binary_loss': binary_loss.detach().item(),
        }
        return loss, logs

    @staticmethod
    def init_params(config, transformer):
        channel = config.get('siconc_channel', 0)
        mean = config.get('siconc_mean')
        std = config.get('siconc_std')
        if mean is None:
            mean = transformer.means[channel].item()
        if std is None:
            std = transformer.stds[channel].item()
        return {
            'masked_loss': config['masked_loss'],
            'kl_weight': config['kl_weight'],
            'siconc_channel': channel,
            'siconc_mean': mean,
            'siconc_std': std,
            'ms_weight': config.get('ms_weight', 0.1),
            'ms_scales': config.get('ms_scales', [1, 2, 4]),
            'ms_scale_weights': config.get('ms_scale_weights', [1.0, 0.5, 0.25]),
            'sobel_weight': config.get('sobel_weight', 0.4),
            'binary_weight': config.get('binary_weight', 0.4),
        }
