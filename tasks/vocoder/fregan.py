import torch.nn.functional as F
from torch import nn
import utils
from modules.fregan.discriminator import ResWiseMultiPeriodDiscriminator, ResWiseMultiScaleDiscriminator
from modules.fregan.generator import FreGAN
from modules.fregan.loss import generator_loss, discriminator_loss
from modules.hifigan.mel_utils import mel_spectrogram
from tasks.vocoder.hifigan import HifiGanTask
from utils.hparams import hparams


class FreGanTask(HifiGanTask):
    def build_model(self):
        self.model_gen = FreGAN(hparams)
        self.model_disc = nn.ModuleDict()
        self.model_disc['mpd'] = ResWiseMultiPeriodDiscriminator()
        self.model_disc['msd'] = ResWiseMultiScaleDiscriminator()
        utils.print_arch(self.model_gen)
        return self.model_gen

    def _training_step(self, sample, batch_idx, optimizer_idx):
        mel = sample['mels']
        y = sample['wavs']
        f0 = sample['f0'] if hparams.get('use_pitch_embed', False) else None
        loss_output = {}
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            y_ = self.model_gen(mel, f0)
            y_mel = mel_spectrogram(y.squeeze(1), hparams).transpose(1, 2)
            y_hat_mel = mel_spectrogram(y_.squeeze(1), hparams).transpose(1, 2)
            loss_output['mel'] = F.l1_loss(y_hat_mel, y_mel) * hparams['lambda_mel']
            _, y_p_hat_g, fmap_f_r, fmap_f_g = self.model_disc['mpd'](y, y_)
            _, y_s_hat_g, fmap_s_r, fmap_s_g = self.model_disc['msd'](y, y_)
            loss_output['a_p'], _ = generator_loss(y_p_hat_g)
            loss_output['a_s'], _ = generator_loss(y_s_hat_g)
            self.y_ = y_.detach()
        else:
            #######################
            #    Discriminator    #
            #######################
            y_ = self.y_
            # MPD
            y_p_hat_r, y_p_hat_g, _, _ = self.model_disc['mpd'](y, y_.detach())
            loss_output['r_p'], loss_output['f_p'], _, _ = discriminator_loss(y_p_hat_r, y_p_hat_g)
            # MSD
            y_s_hat_r, y_s_hat_g, _, _ = self.model_disc['msd'](y, y_.detach())
            loss_output['r_s'], loss_output['f_s'], _, _ = discriminator_loss(y_s_hat_r, y_s_hat_g)
        total_loss = sum(loss_output.values())
        return total_loss, loss_output
