import os
import torch
from torch import nn
import utils
from modules.univnet.discriminator import Discriminator
from modules.univnet.generator import Generator
from modules.univnet.stft_loss import MultiResolutionSTFTLoss
from tasks.vocoder.vocoder_base import VocoderBaseTask
from utils import audio
from utils.hparams import hparams


class UnivNetTask(VocoderBaseTask):
    def build_model(self):
        self.model_gen = Generator(hparams)
        self.model_disc = Discriminator(hparams)
        self.stft_criterion = MultiResolutionSTFTLoss('cuda', hparams['mrd']['resolutions'])

        utils.print_arch(self.model_gen)
        return self.model_gen

    def build_optimizer(self, model):
        optimizer_gen = torch.optim.AdamW(self.model_gen.parameters(), lr=hparams['lr'],
                                          betas=[hparams['adam_b1'], hparams['adam_b2']])
        optimizer_disc = torch.optim.AdamW(self.model_disc.parameters(), lr=hparams['lr'],
                                           betas=[hparams['adam_b1'], hparams['adam_b2']])
        return [optimizer_gen, optimizer_disc]

    def _training_step(self, sample, batch_idx, optimizer_idx):
        mel = sample['mels']
        y = sample['wavs']
        loss_output = {}
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            noise = torch.randn(mel.shape[0], hparams['gen']['noise_dim'], mel.shape[2]).to(mel.device)
            y_ = self.model_gen(mel, noise)
            sc_loss, mag_loss = self.stft_criterion(y_.squeeze(1), y.squeeze(1))
            loss_output['stft_loss'] = (sc_loss + mag_loss) * hparams['stft_lamb']
            res_fake, period_fake = self.model_disc(y_)
            score_loss = 0.0
            for (_, score_fake) in res_fake + period_fake:
                score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
            loss_output['adv'] = score_loss / len(res_fake + period_fake)
            self.y_ = y_.detach()
        else:
            #######################
            #    Discriminator    #
            #######################
            res_real, period_real = self.model_disc(y)
            res_fake, period_fake = self.model_disc(self.y_)
            loss_d = 0.0
            for (_, score_fake), (_, score_real) in zip(res_fake + period_fake, res_real + period_real):
                loss_d += torch.mean(torch.pow(score_real - 1.0, 2))
                loss_d += torch.mean(torch.pow(score_fake, 2))
            loss_output['disc'] = loss_d / len(res_fake + period_fake)
        total_loss = sum(loss_output.values())
        return total_loss, loss_output

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.model_gen.parameters(), hparams['generator_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.model_disc.parameters(), hparams["discriminator_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        pass

    def validation_step(self, sample, batch_idx):
        total_loss, loss_output = self._training_step(sample, batch_idx, 0)
        outputs = {}
        outputs['losses'] = utils.tensors_to_scalars(loss_output)
        outputs['total_loss'] = utils.tensors_to_scalars(total_loss)

        if self.global_step % 50000 == 0 and batch_idx < 10:
            mels = sample['mels']
            y = sample['wavs']
            noise = torch.randn(mels.shape[0], hparams['gen']['noise_dim'], mels.shape[2]).to(mels.device)
            y_ = self.model_gen(mels, noise)
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
        noise = torch.randn(mels.shape[0], hparams['gen']['noise_dim'], mels.shape[2]).to(mels.device)
        loss_output = {}
        y_ = self.model_gen(mels, noise)
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
