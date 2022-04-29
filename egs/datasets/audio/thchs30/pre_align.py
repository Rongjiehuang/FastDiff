from data_gen.tts.vocoder_pre_align import VocoderPreAlign
import os
import glob


class Thchs30PreAlign(VocoderPreAlign):
    def meta_data(self):
        wav_fns = sorted(glob.glob(f'{self.raw_data_dir}/data/*.wav'))
        for wav_fn in wav_fns:
            item_name = os.path.basename(wav_fn)[:-4]
            txt_fn = f'{wav_fn}.trn'
            spk = item_name.split("_")[0]
            yield item_name, wav_fn, (self.load_txt, txt_fn), spk

    @staticmethod
    def load_txt(txt_fn):
        l = open(txt_fn).readlines()[0].strip().replace(" ", "").replace('l=', '')
        return l


if __name__ == "__main__":
    Thchs30PreAlign().process()
