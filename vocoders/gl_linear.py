import librosa
from utils import audio
from utils.audio import griffin_lim
from utils.hparams import hparams, set_hparams
from vocoders.base_vocoder import BaseVocoder, register_vocoder
import numpy as np


@register_vocoder
class GLLinear(BaseVocoder):
    def spec2wav(self, spec, **kwargs):
        phase = kwargs.get('phase', None)
        spec = audio.denormalize(spec, hparams)
        spec = audio.db_to_amp(spec)
        spec = np.abs(spec.T)
        return griffin_lim(spec, hparams, phase)

    @staticmethod
    def wav2spec(wav_fn):
        sample_rate = hparams['audio_sample_rate']
        wav, _ = librosa.core.load(wav_fn, sr=sample_rate)
        fft_size = hparams['fft_size']
        hop_size = hparams['hop_size']
        min_level_db = hparams['min_level_db']
        x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hparams['hop_size'],
                              win_length=hparams['win_size'], window='hann', pad_mode="constant")
        spc = np.abs(x_stft)  # [n_bins, T]
        phase = np.angle(x_stft)
        spc = audio.amp_to_db(spc)
        spc = audio.normalize(spc, {'min_level_db': min_level_db})
        spc = spc.T  # [T, n_bins]

        l_pad, r_pad = audio.librosa_pad_lr(wav, fft_size, hop_size, 1)
        wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
        wav_data = wav[:spc.shape[0] * hop_size]
        return wav_data, spc  # [T, n_bins]


if __name__ == "__main__":
    """
    Run: python vocoders/gl.py --config configs/tts/transformer_tts.yaml
    """
    set_hparams()
    fn = '相爱后动物伤感-07'
    wav_path = f'tmp/{fn}.wav'
    vocoder = GLLinear()
    _, spec = vocoder.wav2spec(wav_path)
    spec, phase = spec[:, :513], spec[:, 513:]
    wav = vocoder.spec2wav(spec.T)
    librosa.output.write_wav(f'tmp/{fn}_gl.wav', wav, 22050)
    wav = vocoder.spec2wav(spec.T, phase=phase.T)
    librosa.output.write_wav(f'tmp/{fn}_gl_phase.wav', wav, 22050)
