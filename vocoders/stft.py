import librosa
from utils import audio
from utils.audio import griffin_lim
from utils.hparams import hparams, set_hparams
from vocoders.base_vocoder import BaseVocoder, register_vocoder
import numpy as np


@register_vocoder
class STFT(BaseVocoder):
    rescale = 100

    def spec2wav(self, spec, **kwargs):
        """

        :param spec: [2, T, n_bins]
        :param kwargs:
        :return: wav
        """
        spec = spec.transpose([0, 2, 1])
        spec = spec[0] + 1j * spec[1]
        spec = spec * STFT.rescale
        return librosa.istft(spec, hop_length=hparams['hop_size'], win_length=hparams['win_size'])

    @staticmethod
    def wav2spec(wav_fn):
        sample_rate = hparams['audio_sample_rate']
        wav, _ = librosa.core.load(wav_fn, sr=sample_rate)
        x_stft = librosa.stft(wav, n_fft=hparams['fft_size'], hop_length=hparams['hop_size'],
                              win_length=hparams['win_size'], window='hann', pad_mode="constant")
        x_stft = x_stft.T / STFT.rescale
        stft = np.abs(x_stft)  # [T, n_bins]
        real = np.real(x_stft)
        imag = np.imag(x_stft)
        real_imag = np.stack([real, imag], -1)  # [T, n_bins, 2]
        return wav, stft, real_imag
