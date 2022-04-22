import librosa

from utils import audio
from utils.hparams import hparams, set_hparams
from utils.pitch_utils import formant_enhancement, to_f0, to_lf0
from vocoders.base_vocoder import BaseVocoder, register_vocoder
import pyworld as pw
import numpy as np


@register_vocoder
class World(BaseVocoder):
    mgc_dim = 60
    beta = 0.1

    def spec2wav(self, inp, **kwargs):
        fs = hparams['audio_sample_rate']
        fft_size = hparams['fft_size']
        beta = World.beta
        frame_period = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
        lf0, mgc, bap = np.split(inp, [1, 1 + World.mgc_dim], 1)
        lf0 = np.ascontiguousarray(lf0).astype(np.double)
        mgc = np.ascontiguousarray(mgc).astype(np.double)
        bap = np.ascontiguousarray(bap).astype(np.double)
        if World.beta != 0:
            formant_enhancement(coded_spectrogram=mgc, beta=beta, fs=fs)
        f0 = to_f0(lf0)
        sp = pw.decode_spectral_envelope(coded_spectral_envelope=mgc, fs=fs, fft_size=fft_size)
        ap = pw.decode_aperiodicity(coded_aperiodicity=bap, fs=fs, fft_size=fft_size)
        y = pw.synthesize(f0, sp, ap, fs, frame_period=frame_period)
        return y

    @staticmethod
    def wav2spec(wav_fn):
        frame_period = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
        fs = hparams['audio_sample_rate']
        x, _ = librosa.core.load(wav_fn, sr=hparams['audio_sample_rate'])
        x = x.astype(np.double)
        _f0, t = pw.dio(x, fs, frame_period=frame_period, f0_floor=80, f0_ceil=750)
        f0 = pw.stonemask(x, _f0, t, fs)
        lf0 = to_lf0(f0)
        sp = pw.cheaptrick(x, f0, t, fs, fft_size=hparams['fft_size'])
        mgc = pw.code_spectral_envelope(sp, fs, World.mgc_dim)
        ap = pw.d4c(x, f0, t, fs, fft_size=hparams['fft_size'])
        bap = pw.code_aperiodicity(ap, fs)
        w_fea = np.concatenate([lf0[:, None], mgc, bap], -1)
        return x, w_fea


if __name__ == "__main__":
    set_hparams(config='configs/tts/lj/fs2.yaml', hparams_str='hop_size=256,fft_size=1024')
    print("| Hparams: ", hparams)
    vocoder = World()
    _, spec = vocoder.wav2spec('tmp/LJ001-0001.wav')
    wav = vocoder.spec2wav(spec)
    if np.abs(wav).max() > 1:
        wav = wav / np.abs(wav).max()
    audio.save_wav(wav, 'tmp/LJ001-0001_out.wav', sr=hparams['audio_sample_rate'])
