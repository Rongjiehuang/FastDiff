import os
import torch
import torch.nn.functional as F
from torch import nn
import utils
from modules.fastspeech.multi_window_disc import Discriminator
from modules.hifigan.hifigan import MultiPeriodDiscriminator, MultiScaleDiscriminator, HifiGanGenerator, feature_loss, \
    generator_loss, discriminator_loss, cond_discriminator_loss
from modules.hifigan.mel_utils import mel_spectrogram
from modules.parallel_wavegan.losses import MultiResolutionSTFTLoss
from tasks.vocoder.vocoder_base import VocoderBaseTask
from utils import audio
from utils.hparams import hparams


class HifiGanTask(VocoderBaseTask):
    def build_model(self):
        self.model_gen = HifiGanGenerator(hparams)
        self.model_disc = nn.ModuleDict()
        self.model_disc['mpd'] = MultiPeriodDiscriminator(use_cond=hparams['use_cond_disc'])
        self.model_disc['msd'] = MultiScaleDiscriminator(use_cond=hparams['use_cond_disc'])
        self.stft_loss = MultiResolutionSTFTLoss()
        if hparams['use_spec_disc']:
            self.model_disc['specd'] = Discriminator(
                time_lengths=[8, 16, 32],
                freq_length=80, hidden_size=128, kernel=(3, 3), cond_size=0, reduction='stack')
        utils.print_arch(self.model_gen)
        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], 'model_gen', 'model_gen', force=True, strict=True)
            self.load_ckpt(hparams['load_ckpt'], 'model_disc', 'model_disc', force=True, strict=True)
        return self.model_gen

    def build_optimizer(self, model):
        optimizer_gen = torch.optim.AdamW(self.model_gen.parameters(),
                                          betas=[hparams['adam_b1'], hparams['adam_b2']],
                                          **hparams['generator_optimizer_params'])
        optimizer_disc = torch.optim.AdamW(self.model_disc.parameters(),
                                           betas=[hparams['adam_b1'], hparams['adam_b2']],
                                           **hparams['discriminator_optimizer_params'])
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return {
            "gen": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer[0],
                **hparams["generator_scheduler_params"]),
            "disc": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer[1],
                **hparams["discriminator_scheduler_params"]),
        }

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
            _, y_p_hat_g, fmap_f_r, fmap_f_g = self.model_disc['mpd'](y, y_, mel)
            _, y_s_hat_g, fmap_s_r, fmap_s_g = self.model_disc['msd'](y, y_, mel)
            loss_output['a_p'] = generator_loss(y_p_hat_g) * hparams['lambda_adv']
            loss_output['a_s'] = generator_loss(y_s_hat_g) * hparams['lambda_adv']
            if hparams['use_fm_loss']:
                loss_output['fm_f'] = feature_loss(fmap_f_r, fmap_f_g)
                loss_output['fm_s'] = feature_loss(fmap_s_r, fmap_s_g)
            if hparams['use_spec_disc']:
                p_ = self.model_disc['specd'](y_hat_mel)['y']
                loss_output['a_mel'] = self.mse_loss_fn(p_, p_.new_ones(p_.size())) * hparams['lambda_mel_adv']
            if hparams['use_ms_stft']:
                loss_output['sc'], loss_output['mag'] = self.stft_loss(y.squeeze(1), y_.squeeze(1))
            self.y_ = y_.detach()
            self.y_mel = y_mel.detach()
            self.y_hat_mel = y_hat_mel.detach()
        else:
            #######################
            #    Discriminator    #
            #######################
            y_ = self.y_
            # MPD
            y_p_hat_r, y_p_hat_g, _, _ = self.model_disc['mpd'](y, y_.detach(), mel)
            loss_output['r_p'], loss_output['f_p'] = discriminator_loss(y_p_hat_r, y_p_hat_g)
            # MSD
            y_s_hat_r, y_s_hat_g, _, _ = self.model_disc['msd'](y, y_.detach(), mel)
            loss_output['r_s'], loss_output['f_s'] = discriminator_loss(y_s_hat_r, y_s_hat_g)
            # spec disc
            if hparams['use_spec_disc']:
                p = self.model_disc['specd'](self.y_mel)['y']
                p_ = self.model_disc['specd'](self.y_hat_mel)['y']
                loss_output["r_mel"] = self.mse_loss_fn(p, p.new_ones(p.size()))
                loss_output["f_mel"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
            if hparams['use_cond_disc']:
                mel_shift = torch.roll(mel, -1, 0)
                yp_f1, yp_f2, _, _ = self.model_disc['mpd'](y.detach(), y_.detach(), mel_shift)
                loss_output["f_p_cd1"] = cond_discriminator_loss(yp_f1)
                loss_output["f_p_cd2"] = cond_discriminator_loss(yp_f2)
                ys_f1, ys_f2, _, _ = self.model_disc['msd'](y.detach(), y_.detach(), mel_shift)
                loss_output["f_s_cd1"] = cond_discriminator_loss(ys_f1)
                loss_output["f_s_cd2"] = cond_discriminator_loss(ys_f2)
        total_loss = sum(loss_output.values())
        return total_loss, loss_output

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.model_gen.parameters(), hparams['generator_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.model_disc.parameters(), hparams["discriminator_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if optimizer_idx == 0:
            self.scheduler['gen'].step(self.global_step // hparams['accumulate_grad_batches'])
        else:
            self.scheduler['disc'].step(self.global_step // hparams['accumulate_grad_batches'])

    def validation_step(self, sample, batch_idx):
        outputs = {}
        total_loss, loss_output = self._training_step(sample, batch_idx, 0)
        outputs['losses'] = utils.tensors_to_scalars(loss_output)
        outputs['total_loss'] = utils.tensors_to_scalars(total_loss)

        if self.global_step % 50000 == 0 and batch_idx < 10:
            mels = sample['mels']
            y = sample['wavs']
            f0 = sample['f0'] if hparams.get('use_pitch_embed', False) else None
            y_ = self.model_gen(mels, f0)
            for idx, (wav_pred, wav_gt, item_name) in enumerate(zip(y_, y, sample["item_name"])):
                wav_pred = wav_pred / wav_pred.abs().max()
                if self.global_step == 50000:
                    wav_gt = wav_gt / wav_gt.abs().max()
                    self.logger.add_audio(f'wav_{batch_idx}_{idx}_gt', wav_gt, self.global_step,
                                          hparams['audio_sample_rate'])
                self.logger.add_audio(f'wav_{batch_idx}_{idx}_pred', wav_pred, self.global_step,
                                      hparams['audio_sample_rate'])
        return outputs

    def test_step(self, sample, batch_idx):
        mels = sample['mels']
        y = sample['wavs']
        f0 = sample['f0'] if hparams.get('use_pitch_embed', False) else None
        loss_output = {}
        y_ = self.model_gen(mels, f0)
        gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(gen_dir, exist_ok=True)
        for idx, (wav_pred, wav_gt, item_name) in enumerate(zip(y_, y, sample["item_name"])):
            wav_gt = wav_gt.clamp(-1, 1)
            wav_pred = wav_pred.clamp(-1, 1)
            audio.save_wav(
                wav_gt.view(-1).cpu().float().numpy(), f'{gen_dir}/{item_name}_gt.wav',
                hparams['audio_sample_rate'])
            audio.save_wav(
                wav_pred.view(-1).cpu().float().numpy(), f'{gen_dir}/{item_name}_pred.wav',
                hparams['audio_sample_rate'])
        return loss_output
