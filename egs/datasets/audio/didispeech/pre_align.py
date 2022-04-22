import os

from data_gen.tts.base_pre_align import BasePreAlign
import glob


class DidiSpeechPreAlign(BasePreAlign):
    def meta_data(self):
        txt_fns = sorted(glob.glob(f'{self.raw_data_dir}/SCRIPT/*.prosody'))
        txts = {}
        for txt_fn in txt_fns:
            spk, _ = os.path.splitext(os.path.basename(txt_fn))
            txts[spk] = {}
            txt_lines = open(txt_fn, encoding='gbk').readlines()[::2]
            for l in txt_lines:
                item_name, txt = l.strip().split('\t')
                item_name = item_name[:17]
                txts[spk][item_name] = txt

        wav_fns = sorted(glob.glob(f'{self.raw_data_dir}/WAV/*/*/*.WAV'))
        for wav_fn in wav_fns:
            item_name = os.path.basename(wav_fn)[:-4]
            spk = wav_fn.split("/")[-3]
            txt = txts[spk][item_name]
            yield item_name, wav_fn, txt, spk


if __name__ == "__main__":
    DidiSpeechPreAlign().process()
