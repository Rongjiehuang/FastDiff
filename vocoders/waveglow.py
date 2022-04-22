from utils.hparams import hparams
from vocoders.base_vocoder import BaseVocoder, register_vocoder


@register_vocoder
class WaveGlow(BaseVocoder):
    def spec2wav(self, mel):
        # TODO waveglow
        pass

    @staticmethod
    def wav2spec(wav_fn):
        from data_gen.tts.data_gen_utils import process_utterance
        wav_data, mel = process_utterance(
            wav_fn, fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'],
            min_level_db=hparams['min_level_db'],
            return_linear=False, vocoder='waveglow')
        mel = mel.T  # [T, 80]
        return mel
