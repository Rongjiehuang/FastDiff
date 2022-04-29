import json
import os

from data_gen.tts.vocoder_pre_align import VocoderPreAlign
import glob


class HifiTTSPreAlign(VocoderPreAlign):
    def meta_data(self):
        meta_fns = sorted(glob.glob(f'{self.raw_data_dir}/hi_fi_tts_v0/*_manifest_*.json'))
        for meta_fn in meta_fns:
            json_lines = open(meta_fn).readlines()
            split = meta_fn.split('_')[-1][:-5]
            spk = os.path.basename(meta_fn).split('_')[0]
            for l in json_lines:
                m = json.loads(l)
                wav_fn = f"{self.raw_data_dir}/hi_fi_tts_v0/{m['audio_filepath']}"
                txt = m['text_normalized']
                item_name = m['audio_filepath'].replace("/", "_")[6:-5]
                yield item_name, wav_fn, txt, spk, {'split': split}


if __name__ == "__main__":
    HifiTTSPreAlign().process()
