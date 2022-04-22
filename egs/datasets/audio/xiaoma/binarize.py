import re
from data_gen.tts.singing.binarize import SingingBinarizer
from utils.hparams import hparams
import pandas as pd


class XiaomaBinarizer(SingingBinarizer):
    """
    Xiaoma binarizer with gender
    """

    def __init__(self):
        super(XiaomaBinarizer, self).__init__()
        gender_csv = 'data/raw/xiaoma1004_long/singer_gender.csv'
        gender_df = pd.read_csv(gender_csv)
        gender_map = {}
        for r_idx, r in gender_df.iterrows():
            gender_map[r['姓名']] = 0 if r['性别'] == '男' else 1
        print("| gender_map: ", gender_map)
        self.item2spk = {}
        self.item2gender = {}
        for item_name in self.item_names:
            gender = None
            singer_name = None
            for dataset in hparams['datasets']:
                if len(re.findall(rf'{dataset}', item_name)) > 0:
                    if '#0514#' in item_name:
                        singer_name = item_name.split("#")[2].split("-")[0]
                        gender = item_name.split("#")[2][0]
                        gender = 0 if gender == '男' else 1
                    else:
                        singer_name = item_name.split("_")[-2]
                        if singer_name not in gender_map:
                            print("| no gender: ", singer_name)
                            continue
                        gender = gender_map[singer_name]
                    break
            assert gender is not None, item_name
            assert singer_name is not None, item_name
            self.item2gender[item_name] = gender
            self.item2spk[item_name] = singer_name

    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            ph = self.item2ph[item_name]
            txt = self.item2txt[item_name]
            tg_fn = self.item2tgfn.get(item_name)
            wav_fn = self.item2wavfn[item_name]
            gender = self.item2gender[item_name]
            spk_id = self.item_name2spk_id(item_name)
            yield item_name, ph, txt, tg_fn, wav_fn, spk_id, gender

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, gender, encoder, binarization_args):
        res = super().process_item(item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args)
        if res is None:
            return None
        res['gender'] = gender
        return res


if __name__ == "__main__":
    XiaomaBinarizer().process()
