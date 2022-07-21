import os

from data_gen.tts.vocoder_pre_align import VocoderPreAlign
import glob
from pathlib import Path

class PreAlign(VocoderPreAlign):
    def meta_data(self):
        wav_fns = sorted(glob.glob(f'{self.raw_data_dir}/*/*/*.wav')) + sorted(glob.glob(f'{self.raw_data_dir}/*/*.wav')) + sorted(glob.glob(f'{self.raw_data_dir}/*.wav'))
        for wav_fn in wav_fns:
            item_name = os.path.basename(wav_fn)[:-4]
            if os.path.exists(wav_fn):
                yield item_name, wav_fn


if __name__ == "__main__":
    PreAlign().process()
