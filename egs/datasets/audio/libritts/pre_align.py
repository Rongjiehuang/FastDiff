import os

from data_gen.tts.vocoder_pre_align import VocoderPreAlign
import glob


class LibrittsPreAlign(VocoderPreAlign):
    def meta_data(self):
        wav_fns = sorted(glob.glob(f'{self.raw_data_dir}/*/*/*.wav'))
        for wav_fn in wav_fns:
            item_name = os.path.basename(wav_fn)[:-4]
            txt = 'Not Needed.'
            spk = item_name.split("_")[0]
            yield item_name, wav_fn, txt, spk


if __name__ == "__main__":
    LibrittsPreAlign().process()
