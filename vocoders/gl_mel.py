import librosa
import numpy as np
from utils.hparams import hparams
from vocoders.base_vocoder import register_vocoder
from vocoders.pwg import PWG
from utils.audio import griffin_lim


@register_vocoder
class GLMel(PWG):
    def __init__(self):
        self.mel_basis = librosa.filters.mel(hparams['audio_sample_rate'], hparams['fft_size'],
                                             hparams['audio_num_mel_bins'], hparams['fmin'], hparams['fmax'])

    def spec2wav(self, spec, **kwargs):
        spec = 10 ** spec
        spec = np.abs(spec)
        x_stft = librosa.util.nnls(self.mel_basis, spec.T)
        return griffin_lim(x_stft, hparams)
