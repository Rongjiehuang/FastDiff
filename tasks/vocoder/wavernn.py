import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DistributedSampler
import utils
from modules.wavernn_modules.utils.distribution import discretized_mix_logistic_loss
from modules.wavernn_modules.utils.dsp import encode_mu_law, float_2_label
from modules.wavernn_modules.wavernn import WaveRNN
from tasks.base_task import BaseTask
from tasks.vocoder.pwg import VocoderDataset, EndlessDistributedSampler
from utils import audio
from utils.hparams import hparams
from tasks.base_task import data_loader


class WaveRNNDataset(VocoderDataset):
    def __init__(self, prefix, shuffle=False):
        hparams['aux_context_window'] = hparams['voc_pad']
        hparams['generator_params'] = {
            'use_pitch_embed': hparams['use_pitch_embed'],
        }
        super().__init__(prefix, shuffle)

    def collater(self, batch):
        ret = super(WaveRNNDataset, self).collater(batch)
        bits = self.hparams['bits']
        wav_np = ret['wavs'].numpy()[:, 0, :]
        ret['wavs_q'] = encode_mu_law(wav_np, mu=2 ** bits) \
            if self.hparams['mu_law'] else float_2_label(wav_np, bits=bits)
        ret['wavs_q'] = torch.LongTensor(ret['wavs_q']).clamp(min=0, max=2 ** bits - 1)
        return ret

class WaveRNNTask(BaseTask):
    def __init__(self):
        super(WaveRNNTask, self).__init__()
        self.dataset_cls = WaveRNNDataset

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls('train', shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_sentences, hparams['endless_ds'])

    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls('valid', shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_valid_sentences)

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls('test', shuffle=False)
        return self.build_dataloader(test_dataset, False, self.max_valid_sentences)

    def build_dataloader(self, dataset, shuffle, max_sentences, endless=False):
        world_size = 1
        rank = 0
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        sampler_cls = DistributedSampler if not endless else EndlessDistributedSampler
        train_sampler = sampler_cls(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            collate_fn=dataset.collater,
            batch_size=max_sentences,
            num_workers=dataset.num_workers,
            sampler=train_sampler,
            pin_memory=True,
        )

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return optimizer

    def build_scheduler(self, optimizer):
        return None

    def build_model(self):
        self.model = WaveRNN(rnn_dims=hparams['voc_rnn_dims'],
                             fc_dims=hparams['voc_fc_dims'],
                             bits=hparams['bits'],
                             pad=hparams['voc_pad'],
                             upsample_factors=hparams['voc_upsample_factors'],
                             feat_dims=hparams['num_mels'],
                             compute_dims=hparams['voc_compute_dims'],
                             res_out_dims=hparams['voc_res_out_dims'],
                             res_blocks=hparams['voc_res_blocks'],
                             hop_length=hparams['hop_size'],
                             sample_rate=hparams['audio_sampling_rate'],
                             mode=hparams['voc_mode'],
                             use_pitch_embed=hparams['use_pitch_embed'])
        return self.model

    def _training_step(self, sample, batch_idx, _):
        mels = sample['mels']
        y = sample['wavs'][:, 0]
        p = sample['pitches']
        y_ = self.model(y, mels, p)
        if hparams['voc_mode'] == 'RAW':
            y = sample['wavs_q']
            loss = F.cross_entropy(y_[:, :-1].transpose(1, 2), y[:, 1:])
        else:
            loss = discretized_mix_logistic_loss(y_[:, :-1], y[:, 1:, None])
        return loss, {'loss': loss}

    def validation_step(self, sample, batch_idx):
        loss = self._training_step(sample, batch_idx, None)[0]
        mels = sample['mels']
        outputs = {
            'total_loss': loss,
            'nsamples': mels.shape[0]
        }
        return utils.tensors_to_scalars(outputs)

    def test_step(self, sample, batch_idx):
        mels = sample['mels']
        y = sample['wavs'][:, 0]
        p = sample['pitches']
        y_ = self.model.generate(mels, p, batched=True, target=12800, overlap=2048)
        gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(gen_dir, exist_ok=True)
        for idx, (wav_pred, wav_gt) in enumerate(zip(y_, y)):
            wav_gt = wav_gt / wav_gt.abs().max()
            wav_pred = wav_pred / wav_pred.abs().max()
            audio.save_wav(wav_gt.view(-1).cpu().float().numpy(), f'{gen_dir}/wav_{batch_idx}_{idx}_gt.wav', 22050)
            audio.save_wav(wav_pred.view(-1).cpu().float().numpy(), f'{gen_dir}/wav_{batch_idx}_{idx}_pred.wav', 22050)
        return {}
