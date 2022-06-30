import os

from data_gen.tts.vocoder_pre_align import VocoderPreAlign
import glob


class VCTKPreAlign(VocoderPreAlign):
    def meta_data(self):
        wav_fns = glob.glob(f'{self.raw_data_dir}/wav48/*/*.wav')
        for wav_fn in wav_fns:
            item_name = os.path.basename(wav_fn)[:-4]
            spk = item_name.split("_")[0]
            txt = "Not Needed"
            if os.path.exists(wav_fn):
                yield item_name, wav_fn, txt, spk


if __name__ == "__main__":
    VCTKPreAlign().process()
