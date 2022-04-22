from data_gen.tts.base_pre_align import BasePreAlign
import pandas as pd


class BlzPreAlign(BasePreAlign):
    def meta_data(self):
        df = pd.read_csv(f'{self.raw_data_dir}/meta_data.csv')
        for r_idx, row in df.iterrows():
            item_name = row['wav']
            txt = row['txt']
            wav_fn = f"{self.raw_data_dir}/wavs/{item_name}.wav"
            yield item_name, wav_fn, txt, 'SPK1'


if __name__ == "__main__":
    BlzPreAlign().process()
