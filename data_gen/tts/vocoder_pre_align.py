import os

os.environ["OMP_NUM_THREADS"] = "1"

import librosa
from utils import audio
from data_gen.tts.data_gen_utils import is_sil_phoneme
from utils.multiprocess_utils import chunked_multiprocess_run
import traceback
import importlib
from utils.hparams import hparams, set_hparams
import json
import os
import subprocess
from tqdm import tqdm
import pandas as pd
from utils.rnnoise import rnnoise


class VocoderPreAlign:
    def __init__(self):
        self.pre_align_args = hparams['pre_align_args']
        self.raw_data_dir = hparams['raw_data_dir']
        self.processed_dir = hparams['processed_data_dir']

    def meta_data(self):
        raise NotImplementedError


    @staticmethod
    def process_wav(idx, item_name, wav_fn, processed_dir, pre_align_args):
        if pre_align_args['sox_to_wav'] or pre_align_args['trim_sil'] or \
                pre_align_args['sox_resample'] or pre_align_args['denoise']:
            sr = hparams['audio_sample_rate']
            new_wav_fn = f"{processed_dir}/wav_inputs/{idx}"
            subprocess.check_call(f'sox "{wav_fn}" -t wav "{new_wav_fn}.wav"', shell=True)
            if pre_align_args['sox_resample']:
                subprocess.check_call(f'sox -v 0.95 "{new_wav_fn}.wav" -r{sr} "{new_wav_fn}_rs.wav"', shell=True)
                new_wav_fn = new_wav_fn + '_rs'
            if pre_align_args['denoise']:
                rnnoise(wav_fn, new_wav_fn + '_denoise.wav', out_sample_rate=sr)
                new_wav_fn = new_wav_fn + '_denoise'
            if pre_align_args['trim_sil']:
                y, _ = librosa.core.load(new_wav_fn + '.wav', sr=sr)
                y, _ = librosa.effects.trim(y)
                audio.save_wav(y, new_wav_fn + '_trim.wav', sr, norm=True)
                new_wav_fn = new_wav_fn + '_trim'
            return new_wav_fn + '.wav'
        else:
            return wav_fn

    def process(self):
        set_hparams()
        processed_dir = self.processed_dir
        subprocess.check_call(f'rm -rf {processed_dir}/mfa_inputs', shell=True)
        os.makedirs(f"{processed_dir}/wav_inputs", exist_ok=True)
        meta_df = []

        args = []
        meta_data = []
        for idx, inp_args in enumerate(tqdm(self.meta_data(), desc='Load meta data')):
            meta_data.append(inp_args)
            item_name, wav_fn = inp_args
            args.append([
                idx, item_name, wav_fn, processed_dir, self.pre_align_args
            ])
        item_names = [x[1] for x in args]
        assert len(item_names) == len(set(item_names)), 'Key `item_name` should be Unique.'

        for inp_args, res in zip(tqdm(meta_data, 'Processing'), chunked_multiprocess_run(self.process_job, args)):
            item_name, wav_fn = inp_args
            if res is None:
                print(f"| Skip {wav_fn}.")
                continue
            wav_fn = res
            meta_df.append({'item_name': item_name, 'wav_fn': wav_fn})


        # save to csv
        meta_df = pd.DataFrame(meta_df)
        meta_df.to_csv(f"{processed_dir}/metadata_phone.csv")

    @classmethod
    def process_job(cls, idx, item_name, wav_fn, processed_dir, pre_align_args):
        try:
            wav_fn = cls.process_wav(idx, item_name, wav_fn, processed_dir, pre_align_args)
            if wav_fn is None:
                return None
        except:
            traceback.print_exc()
            return None
        group = idx // pre_align_args['nsample_per_mfa_group']  # group MFA inputs for better parallelism
        os.makedirs(f'{processed_dir}/mfa_inputs/{group}', exist_ok=True)
        ext = os.path.splitext(wav_fn)[1]
        new_wav_fn = f"{processed_dir}/mfa_inputs/{group}/{idx:07d}_{item_name}{ext}"
        cp_cmd = 'mv' if 'wav_inputs' in wav_fn else 'cp'
        subprocess.check_call(f'{cp_cmd} "{wav_fn}" "{new_wav_fn}"', shell=True)
        wav_fn = new_wav_fn
        return wav_fn
