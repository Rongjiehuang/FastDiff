import os

import numpy as np
import torch

import utils
from modules.parallel_wavegan.models import MelGANGenerator, \
    MelGANMultiScaleDiscriminator
from tasks.vocoder.pwg import PwgTask
from utils import audio
from utils.hparams import hparams


class MelGANTask(PwgTask):
    def build_model(self):
        self.model_gen = MelGANGenerator(**hparams["generator_params"])
        self.model_disc = MelGANMultiScaleDiscriminator(**hparams["discriminator_params"])
        self.disc_params = list(self.model_disc.parameters())
        parameters = filter(lambda p: p.requires_grad, self.model_gen.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        utils.print_arch(self.model_gen)
        return parameters

    def build_optimizer(self, model):
        optimizer_gen = torch.optim.Adam(self.model_gen.parameters(),
                                         **hparams["generator_optimizer_params"])
        optimizer_disc = torch.optim.Adam(self.disc_params,
                                          **hparams["discriminator_optimizer_params"])
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return {
            "gen": torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer[0],
                **hparams["generator_scheduler_params"]),
            "disc": torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer[1],
                **hparams["discriminator_scheduler_params"]),
        }

    def _training_step(self, sample, batch_idx, optimizer_idx):
        mels = sample['mels']
        y = sample['wavs']
        pitch = sample['pitches']
        y = y.squeeze(1)
        loss_output = {}
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            # calculate generator loss
            y_ = self.model_gen(mels, f0=sample.get('f0'), pitch=pitch).squeeze(1)
            sc_loss, mag_loss = self.stft_loss(y_, y)
            total_loss = sc_loss + mag_loss
            loss_output['sc'] = sc_loss
            loss_output['mag'] = mag_loss
            if self.global_step > hparams["disc_start_steps"]:
                p_ = self.model_disc(y_.unsqueeze(1))
                # for multi-scale discriminator
                adv_loss = 0.0
                for i in range(len(p_)):
                    adv_loss += self.mse_loss_fn(p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size()))
                adv_loss /= (i + 1)
                loss_output['a'] = adv_loss

                # feature matching loss
                if hparams["use_feat_match_loss"]:
                    # no need to track gradients
                    with torch.no_grad():
                        p = self.model_disc(y.unsqueeze(1))
                    fm_loss = 0.0
                    for i in range(len(p_)):
                        for j in range(len(p_[i]) - 1):
                            fm_loss += self.l1_loss_fn(p_[i][j], p[i][j].detach())
                    fm_loss /= (i + 1) * (j + 1)
                    loss_output["fm"] = fm_loss
                    adv_loss += hparams["lambda_feat_match"] * fm_loss
                total_loss += hparams["lambda_adv"] * adv_loss
                self.y_pred = y_.detach()
        else:
            #######################
            #    Discriminator    #
            #######################
            if self.global_step > hparams["disc_start_steps"]:
                y_pred = self.model_gen(mels, f0=sample.get('f0'), pitch=pitch).squeeze(1)
                # calculate discriminator loss
                p = self.model_disc(y.unsqueeze(1))
                p_ = self.model_disc(y_pred.unsqueeze(1))
                # for multi-scale discriminator
                real_loss = 0.0
                fake_loss = 0.0
                for i in range(len(p)):
                    real_loss += self.mse_loss_fn(p[i][-1], p[i][-1].new_ones(p[i][-1].size()))
                    fake_loss += self.mse_loss_fn(p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size()))
                real_loss /= (i + 1)
                fake_loss /= (i + 1)
                loss_output["r"] = real_loss
                loss_output["f"] = fake_loss
                total_loss = real_loss + fake_loss
            else:
                # skip disc training
                return None
        return total_loss, loss_output

    def validation_step(self, sample, batch_idx):
        mels = sample['mels']
        y = sample['wavs']
        pitch = sample['pitches']
        loss_output = {}

        #######################
        #      Generator      #
        #######################
        y_ = self.model_gen(mels, f0=sample.get('f0'), pitch=pitch)
        y, y_ = y.squeeze(1), y_.squeeze(1)
        sc_loss, mag_loss = self.stft_loss(y_, y)
        # add to total eval loss
        loss_output["sc"] = sc_loss
        loss_output["mag"] = mag_loss
        loss_output = utils.tensors_to_scalars(loss_output)
        return loss_output

    def test_step(self, sample, batch_idx):
        mels = sample['mels']
        y = sample['wavs']
        pitch = sample['pitches']
        loss_output = {}
        y_ = self.model_gen(mels, f0=sample.get('f0'), pitch=pitch)
        gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(gen_dir, exist_ok=True)
        for idx, (wav_pred, wav_gt, item_name) in enumerate(zip(y_, y, sample["item_name"])):
            wav_gt = wav_gt / wav_gt.abs().max()
            wav_pred = wav_pred / wav_pred.abs().max()
            audio.save_wav(wav_gt.view(-1).cpu().float().numpy(), f'{gen_dir}/{item_name}_gt.wav', 22050)
            audio.save_wav(wav_pred.view(-1).cpu().float().numpy(), f'{gen_dir}/{item_name}_pred.wav', 22050)
        return loss_output
