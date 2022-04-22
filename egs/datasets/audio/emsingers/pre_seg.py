import os
import subprocess
from data_gen.tts.singing.pre_seg import SingingPreSegBinarizer, SingingPreSegPreAlign
from data_gen.tts.data_gen_utils import trim_long_silences
from utils.audio import save_wav, to_mp3
from utils.hparams import set_hparams
from utils.rnnoise import rnnoise


class EmsingerSingingPreSegPreAlign(SingingPreSegPreAlign):
    @staticmethod
    def process_wav(idx, item_name, wav_fn, processed_dir, pre_align_args):
        out_path = f"{processed_dir}/wav_inputs/{item_name}"
        os.makedirs(f"{processed_dir}/wav_inputs", exist_ok=True)
        singer, dir_name, song_name = item_name.split("#")
        hparams = pre_align_args['hparams']
        if os.path.exists(f'{out_path}.mp3'):
            return f'{out_path}.mp3'

        if singer in hparams['short_datasets']:
            assert os.path.exists(wav_fn)
            subprocess.check_call(f'cp "{wav_fn}" "{out_path}.wav"', shell=True)
        else:
            wav, mask, sr = trim_long_silences(wav_fn, sr=None, return_raw_wav=True)
            noise_dur = sum(mask == 0) / sr
            save_wav(wav[mask], f"{out_path}.wav", sr, False)
            if singer in hparams['denoise_datasets'] and noise_dur > 1:
                rnnoise(f"{out_path}.wav", f"{out_path}.wav")
        to_mp3(out_path)
        return f'{out_path}.mp3'


if __name__ == "__main__":
    set_hparams()
    EmsingerSingingPreSegPreAlign().process()
    SingingPreSegBinarizer().process()
