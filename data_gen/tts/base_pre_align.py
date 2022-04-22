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


class BasePreAlign:
    def __init__(self):
        self.pre_align_args = hparams['pre_align_args']
        txt_processor = self.pre_align_args['txt_processor']
        self.txt_processor = importlib.import_module(f'data_gen.tts.txt_processors.{txt_processor}').TxtProcessor
        self.raw_data_dir = hparams['raw_data_dir']
        self.processed_dir = hparams['processed_data_dir']

    def meta_data(self):
        raise NotImplementedError

    @staticmethod
    def load_txt(txt_fn):
        l = open(txt_fn).readlines()[0].strip()
        return l

    @staticmethod
    def process_wav(idx, item_name, wav_fn, processed_dir, pre_align_args):
        if pre_align_args['sox_to_wav'] or pre_align_args['trim_sil'] or \
                pre_align_args['sox_resample'] or pre_align_args['denoise']:
            sr = hparams['audio_sample_rate']
            new_wav_fn = f"{processed_dir}/wav_inputs/{idx}"
            subprocess.check_call(f'sox "{wav_fn}" -t wav "{new_wav_fn}.wav"', shell=True)
            if pre_align_args['trim_sil']:
                y, sr = librosa.core.load(new_wav_fn + '.wav')
                y, _ = librosa.effects.trim(y)
                audio.save_wav(y, new_wav_fn + '_trim.wav', sr)
                new_wav_fn = new_wav_fn + '_trim'
            if pre_align_args['sox_resample']:
                subprocess.check_call(f'sox -v 0.95 "{new_wav_fn}.wav" -r{sr} "{new_wav_fn}_rs.wav"', shell=True)
                new_wav_fn = new_wav_fn + '_rs'
            if pre_align_args['denoise']:
                rnnoise(wav_fn, new_wav_fn + '_denoise.wav', out_sample_rate=sr)
                new_wav_fn = new_wav_fn + '_denoise'
            return new_wav_fn + '.wav'
        else:
            return wav_fn

    def process(self):
        set_hparams()
        processed_dir = self.processed_dir
        subprocess.check_call(f'rm -rf {processed_dir}/mfa_inputs', shell=True)
        os.makedirs(f"{processed_dir}/wav_inputs", exist_ok=True)
        phone_set = set()
        meta_df = []
        word_level_dict = set()

        args = []
        meta_data = []
        for idx, inp_args in enumerate(tqdm(self.meta_data(), desc='Load meta data')):
            if len(inp_args) == 4:
                inp_args = [*inp_args, {}]
            meta_data.append(inp_args)
            item_name, wav_fn, txt_or_fn, spk, others = inp_args
            args.append([
                idx, item_name, self.txt_processor, txt_or_fn, wav_fn, others, processed_dir, self.pre_align_args
            ])
        item_names = [x[1] for x in args]
        assert len(item_names) == len(set(item_names)), 'Key `item_name` should be Unique.'

        for inp_args, res in zip(tqdm(meta_data, 'Processing'), chunked_multiprocess_run(self.process_job, args)):
            item_name, wav_fn, txt_or_fn, spk, others = inp_args
            if res is None:
                print(f"| Skip {wav_fn}.")
                continue
            phs, phs_for_dict, phs_for_align, txt, txt_raw, wav_fn = res
            meta_df.append({
                'item_name': item_name, 'spk': spk, 'txt': txt, 'txt_raw': txt_raw,
                'ph': phs, 'wav_fn': wav_fn, 'others': json.dumps(others)})
            for ph in phs.split(" "):
                phone_set.add(ph)
            phs_for_dict.add('SIL')
            for t in phs_for_dict:
                word_level_dict.add(f"{t.replace(' ', '_')} {t}")
        phone_set = sorted(phone_set)
        word_level_dict = sorted(word_level_dict)
        print("| phone_set[:200]: ", phone_set[:200])
        with open(f'{processed_dir}/dict.txt', 'w') as f:
            for ph in phone_set:
                f.write(f'{ph} {ph}\n')
        json.dump(phone_set, open(f'{processed_dir}/phone_set.json', 'w'))

        # save to csv
        meta_df = pd.DataFrame(meta_df)
        meta_df.to_csv(f"{processed_dir}/metadata_phone.csv")
        with open(f'{processed_dir}/mfa_dict.txt', 'w') as f:
            for l in word_level_dict:
                f.write(f'{l}\n')

    @staticmethod
    def process_text(txt_processor, txt_raw, pre_align_args):
        phs, txt = txt_processor.process(txt_raw, pre_align_args)
        phs = [p.strip() for p in phs if p.strip() != ""]

        # remove sil phoneme in head and tail
        while len(phs) > 0 and is_sil_phoneme(phs[0]):
            phs = phs[1:]
        while len(phs) > 0 and is_sil_phoneme(phs[-1]):
            phs = phs[:-1]
        phs = ["<BOS>"] + phs + ["<EOS>"]
        phs_ = []
        for i in range(len(phs)):
            if len(phs_) == 0 or not is_sil_phoneme(phs[i]) or not is_sil_phoneme(phs_[-1]):
                phs_.append(phs[i])
            elif phs_[-1] == '|' and is_sil_phoneme(phs[i]) and phs[i] != '|':
                phs_[-1] = phs[i]
        cur_word = []
        phs_for_align = []
        phs_for_dict = set()
        for p in phs_:
            if is_sil_phoneme(p):
                if len(cur_word) > 0:
                    phs_for_align.append('_'.join(cur_word))
                    phs_for_dict.add(' '.join(cur_word))
                    cur_word = []
                if p not in txt_processor.sp_phonemes():
                    phs_for_align.append('SIL')
            else:
                cur_word.append(p)
        phs = " ".join(phs_)
        phs_for_align = " ".join(phs_for_align)
        return phs, phs_for_dict, phs_for_align, txt

    @classmethod
    def process_job(cls, idx, item_name, g2p_func, txt_or_fn, wav_fn, others, processed_dir, pre_align_args):
        try:
            if isinstance(txt_or_fn, list) or isinstance(txt_or_fn, tuple):
                txt_load_func, txt_load_func_args = txt_or_fn
                txt_raw = txt_load_func(txt_load_func_args)
            else:
                txt_raw = txt_or_fn
        except Exception as e:
            if not pre_align_args['allow_no_txt']:
                raise e
            else:
                txt_raw = 'NO_TEXT'
        if txt_raw is None:
            return None
        try:
            phs, phs_for_dict, phs_for_align, txt = cls.process_text(g2p_func, txt_raw, pre_align_args)
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
        with open(f'{processed_dir}/mfa_inputs/{group}/{idx:07d}_{item_name}.lab', 'w') as f_txt:
            f_txt.write(phs_for_align)
        wav_fn = new_wav_fn
        return phs, phs_for_dict, phs_for_align, txt, txt_raw, wav_fn
