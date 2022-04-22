import os

import torch

import utils
from modules.wavenet_vocoder.wavenet import WaveNet
from modules.wavernn_modules.utils.distribution import discretized_mix_logistic_loss
from tasks.vocoder.wavernn import WaveRNNTask
from utils import audio
from utils.hparams import hparams


class WaveNetTask(WaveRNNTask):
    def __init__(self):
        super().__init__()
        self.loss_func = discretized_mix_logistic_loss
        self.max_valid_sentences = hparams['max_valid_sentences']
        self.max_sentences = hparams['max_sentences']

    def build_model(self):
        model = WaveNet(
            out_channels=hparams['out_channels'],
            layers=hparams['layers'],
            stacks=hparams['stacks'],
            residual_channels=hparams['residual_channels'],
            gate_channels=hparams['gate_channels'],
            skip_out_channels=hparams['skip_out_channels'],
            cin_channels=hparams['cin_channels'],
            gin_channels=hparams['gin_channels'],
            n_speakers=hparams['n_speakers'],
            dropout=hparams['dropout'],
            kernel_size=hparams['kernel_size'],
            cin_pad=hparams['cin_pad'],
            upsample_conditional_features=hparams['upsample_conditional_features'],
            upsample_params=hparams['upsample_params'],
            output_distribution=hparams['output_distribution'],
            scalar_input=True,
            use_pitch_embed=hparams['use_pitch_embed'],
        )
        utils.print_arch(model)
        return model

    def _training_step(self, sample, batch_idx, _):
        p = sample['pitches']
        mels = sample['mels']
        y = sample['wavs']
        y_ = self.model(y, mels, p=p)
        loss = self.loss_func(y_[:, :, :-1].transpose(1, 2), y[:, :, 1:].transpose(1, 2))
        return loss, {'loss': loss}

    def test_step(self, sample, batch_idx):
        mels = sample['mels']
        y = sample['wavs'][:, 0]
        p = sample['pitches']
        y_ = self.model.generate(mels, p)
        gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(gen_dir, exist_ok=True)
        sample_rate = hparams['audio_sample_rate']
        for idx, (wav_pred, wav_gt) in enumerate(zip(y_, y)):
            wav_gt = wav_gt / wav_gt.abs().max()
            wav_pred = wav_pred / wav_pred.abs().max()
            audio.save_wav(wav_gt.view(-1).cpu().float().numpy(),
                           f'{gen_dir}/wav_{batch_idx}_{idx}_gt.wav', sample_rate)
            audio.save_wav(wav_pred.view(-1).cpu().float().numpy(),
                           f'{gen_dir}/wav_{batch_idx}_{idx}_pred.wav', sample_rate)
        return {}

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=200000, gamma=0.5)
