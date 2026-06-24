# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.autograd as autograd

from lada.models.basicvsrpp.mmagic.registry import MODELS
from lada.models.basicvsrpp.mmagic.real_basicvsr import RealBasicVSR


@MODELS.register_module()
class RVRTGan(RealBasicVSR):
    """RVRT GAN training model with R1 gradient penalty.

    Uses RVRTNet/RVRTGanNet as generator with ProjectedDiscriminator
    + hinge loss + R1 regularization.

    Args:
        generator (dict): Config for RVRTNet or RVRTGanNet.
        discriminator (dict, optional): Config for discriminator.
        gan_loss (dict, optional): Config for GAN loss.
        pixel_loss (dict, optional): Config for pixel loss.
        perceptual_loss (dict, optional): Config for perceptual loss.
        is_use_ema (bool): Use EMA on generator. Default: False.
        r1_weight (float): R1 gradient penalty weight. Default: 0.0
            (disabled). Recommended: 1.0 with hinge loss.
        r1_interval (int): Compute R1 every N steps. Default: 1.
        train_cfg (dict): Training config.
        test_cfg (dict): Testing config.
        init_cfg (dict): Initialization config.
        data_preprocessor (dict): Data preprocessor config.
    """

    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_loss=None,
                 pixel_loss=None,
                 perceptual_loss=None,
                 is_use_ema=False,
                 r1_weight=0.0,
                 r1_interval=1,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None):

        super().__init__(
            generator=generator,
            discriminator=discriminator,
            gan_loss=gan_loss,
            pixel_loss=pixel_loss,
            perceptual_loss=perceptual_loss,
            is_use_sharpened_gt_in_pixel=False,
            is_use_sharpened_gt_in_percep=False,
            is_use_sharpened_gt_in_gan=False,
            is_use_ema=is_use_ema,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.r1_weight = r1_weight
        self.r1_interval = r1_interval

    def d_step_with_optim(self, batch_outputs, batch_gt_data, optim_wrapper):
        """D step with optional R1 gradient penalty.

        Overrides RealBasicVSR to add R1 gradient penalty computed
        on real data for improved training stability.
        """
        log_vars = dict()
        d_optim_wrapper = optim_wrapper['discriminator']

        with d_optim_wrapper.optim_context(self):
            loss_d_real = self.d_step_real(batch_outputs, batch_gt_data)

        # R1 gradient penalty on real data
        if self.r1_weight > 0 and self.step_counter % self.r1_interval == 0:
            gt_gan = batch_gt_data[2].detach().requires_grad_(True)
            real_pred = self.discriminator(gt_gan)
            grad_real = autograd.grad(
                outputs=real_pred.sum(),
                inputs=gt_gan,
                create_graph=True,
                only_inputs=True)[0]
            r1_penalty = grad_real.pow(2).sum(dim=[1, 2, 3]).mean() * (
                self.r1_weight * 0.5)
            loss_d_real = loss_d_real + r1_penalty
            log_vars['loss_r1'] = r1_penalty

        parsed_losses_dr, log_vars_dr = self.parse_losses(
            dict(loss_d_real=loss_d_real))
        log_vars.update(log_vars_dr)
        loss_dr = d_optim_wrapper.scale_loss(parsed_losses_dr)
        d_optim_wrapper.backward(loss_dr)

        with d_optim_wrapper.optim_context(self):
            loss_d_fake = self.d_step_fake(batch_outputs, batch_gt_data)

        parsed_losses_df, log_vars_df = self.parse_losses(
            dict(loss_d_fake=loss_d_fake))
        log_vars.update(log_vars_df)
        loss_df = d_optim_wrapper.scale_loss(parsed_losses_df)
        d_optim_wrapper.backward(loss_df)

        if d_optim_wrapper.should_update():
            d_optim_wrapper.step()
            d_optim_wrapper.zero_grad()

        return log_vars

    def extract_gt_data(self, data_samples):
        """Extract GT data as (pixel, perceptual, gan) triple."""
        gt = data_samples.gt_img
        gt_pixel, gt_percep, gt_gan = gt.clone(), gt.clone(), gt.clone()
        n, t, c, h, w = gt_pixel.size()
        gt_pixel = gt_pixel.view(-1, c, h, w)
        gt_percep = gt_percep.view(-1, c, h, w)
        gt_gan = gt_gan.view(-1, c, h, w)
        return gt_pixel, gt_percep, gt_gan
