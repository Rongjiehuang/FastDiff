from modules.parallel_wavegan.layers import PQMF
from modules.waveglow.waveglow import WaveGlow
from modules.waveglow.common_layers import WaveGlowLoss
import torch
import utils
from tasks.vocoder.pwg import PwgTask
from utils import audio
from utils.hparams import hparams
import os


class WaveGlowTask(PwgTask):
    def __init__(self):
        super(WaveGlowTask, self).__init__()
        self.criterion = WaveGlowLoss()
        self.pqmf = PQMF()

    def build_model(self):
        self.model = WaveGlow(**hparams['waveglow_config'])
        utils.print_arch(self.model)
        return self.model

    def run_model(self, sample):
        mel = sample['mels']
        y = sample['wavs']
        y_sb = self.pqmf.analysis(y)
        output = self.model(mel, y_sb)
        loss = self.criterion(output)
        loss_log = {'loss': loss.item()}
        return loss, loss_log

    def _training_step(self, sample, batch_idx, _):
        return self.run_model(sample)

    def validation_step(self, sample, batch_idx):
        loss, loss_log = self.run_model(sample)
        if self.global_step % 50000 == 0 and batch_idx < 10:
            mels = sample['mels']
            y_ = self.model.infer(mels)
            y_ = self.pqmf.synthesis(y_)
            y = sample['wavs']

            for idx, (wav_pred, wav_gt, item_name) in enumerate(zip(y_, y, sample["item_name"])):
                wav_pred = wav_pred / wav_pred.abs().max()
                if self.global_step == 50000:
                    wav_gt = wav_gt / wav_gt.abs().max()
                    self.logger.add_audio(f'wav_{batch_idx}_{idx}_gt', wav_gt, self.global_step,
                                          hparams['audio_sample_rate'])
                self.logger.add_audio(f'wav_{batch_idx}_{idx}_pred', wav_pred, self.global_step,
                                      hparams['audio_sample_rate'])
        return {'total_loss': loss.item(), 'nsamples': len(sample['wavs'])}

    def test_step(self, sample, batch_idx):
        mels = sample['mels']
        y = sample['wavs']
        loss_output = {}
        y_ = self.model.infer(mels)
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

    def test_end(self, self_outputs):
        return {}

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=float(hparams['lr']))
        return optimizer

    def build_scheduler(self, optimizer):
        self.scheduler = None  # According to NVIDIA waveglow repo

    def on_before_optimization(self, opt_idx):
        pass

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        pass
