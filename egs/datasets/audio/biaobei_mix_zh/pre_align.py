from data_gen.tts.vocoder_pre_align import VocoderPreAlign
import re


class BiaobeiMixZhPreAlign(VocoderPreAlign):
    def meta_data(self):
        input_dir = self.raw_data_dir
        with open(f"{input_dir}/中文女生DB-6/CN/text.txt", encoding='utf-8') as f:
            bb_lines = f.readlines()[::2]
        for l_idx, l in (enumerate([re.sub("\#\d+", "", l.split('\t')[1].strip()) for l in bb_lines])):
            item_name = f'{l_idx + 1:06d}'
            wav_fn = f"{input_dir}/中文女生DB-6/CN/wav/{l_idx + 1:06d}.wav"
            yield item_name, wav_fn, l, 'SPK1'


if __name__ == "__main__":
    BiaobeiMixZhPreAlign().process()
