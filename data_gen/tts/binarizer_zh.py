import json
import os
from collections import Counter

os.environ["OMP_NUM_THREADS"] = "1"

from data_gen.tts.base_binarizer import BaseBinarizer, BinarizationError
from data_gen.tts.data_gen_utils import get_mel2ph, PUNCS
from utils.hparams import set_hparams, hparams
import numpy as np


class ZhBinarizer(BaseBinarizer):
    def _word_encoder(self):
        fn = f"{hparams['binary_data_dir']}/word_set.json"
        if self.binarization_args['reset_word_dict']:
            word_set = []
            for word_sent in self.item2txt.values():
                word_set += list(word_sent)
            word_set = Counter(word_set)
            total_words = sum(word_set.values())
            word_set = word_set.most_common(hparams['word_size'])
            num_unk_words = total_words - sum([x[1] for x in word_set])
            word_set = [x[0] for x in word_set]
            json.dump(word_set, open(fn, 'w'))
            print(f"| #total words: {total_words}, #unk_words: {num_unk_words}")
        else:
            word_set = json.load(open(fn, 'r'))
        print("| Word dict size: ", len(word_set), word_set[:10])
        from utils.text_encoder import TokenTextEncoder
        return TokenTextEncoder(None, vocab_list=word_set, replace_oov='<UNK>')

    @staticmethod
    def get_align(tg_fn, res):
        ph = res['ph']
        mel = res['mel']
        phone = res['phone']
        if tg_fn is not None and os.path.exists(tg_fn):
            _, dur = get_mel2ph(tg_fn, ph, mel, hparams)
        else:
            raise BinarizationError(f"Align not found")
        ph_list = ph.split(" ")
        assert len(dur) == len(ph_list)
        mel2ph = []
        for i in range(len(dur)):
            mel2ph += [i + 1] * dur[i]
        mel2ph = np.array(mel2ph)
        if mel2ph.max() - 1 >= len(phone):
            raise BinarizationError(f"| Align does not match: {(mel2ph.max() - 1, len(phone))}")
        res['mel2ph'] = mel2ph
        res['dur'] = dur

        # char-level pitch
        if 'f0' in res:
            res['f0_ph'] = np.array([0 for _ in res['f0']], dtype=float)
            char_start_idx = 0
            f0s_char = []
            # ph_list = 0
            for idx, (f0_, ph_idx) in enumerate(zip(res['f0'], res['mel2ph'])):
                is_pinyin = ph_list[ph_idx - 1][0].isalpha()
                if not is_pinyin or ph_idx - res['mel2ph'][idx - 1] > 1:
                    if len(f0s_char) > 0:
                        res['f0_ph'][char_start_idx:idx] = sum(f0s_char) / len(f0s_char)
                    f0s_char = []
                    char_start_idx = idx
                    if not is_pinyin:
                        char_start_idx += 1
                if f0_ > 0:
                    f0s_char.append(f0_)

    @staticmethod
    def get_word(res, word_encoder):
        ph_split = res['ph'].split(" ")
        # ph side mapping to word
        ph_words = []  # ['<BOS>', 'N_AW1_', ',', 'AE1_Z_|', 'AO1_L_|', 'B_UH1_K_S_|', 'N_AA1_T_|', ....]
        ph2word = np.zeros([len(ph_split)], dtype=int)
        last_ph_idx_for_word = []  # [2, 11, ...]
        for i, ph in enumerate(ph_split):
            if ph in ['|', '#']:
                last_ph_idx_for_word.append(i)
            elif not ph[0].isalnum():
                if ph not in ['<BOS>']:
                    last_ph_idx_for_word.append(i - 1)
                last_ph_idx_for_word.append(i)
        start_ph_idx_for_word = [0] + [i + 1 for i in last_ph_idx_for_word[:-1]]
        for i, (s_w, e_w) in enumerate(zip(start_ph_idx_for_word, last_ph_idx_for_word)):
            ph_words.append(ph_split[s_w:e_w + 1])
            ph2word[s_w:e_w + 1] = i
        ph2word = ph2word.tolist()
        ph_words = ["_".join(w) for w in ph_words]

        # mel side mapping to word
        mel2word = []
        dur_word = [0 for _ in range(len(ph_words))]
        for i, m2p in enumerate(res['mel2ph']):
            word_idx = ph2word[m2p - 1]
            mel2word.append(ph2word[m2p - 1])
            dur_word[word_idx] += 1
        ph2word = [x + 1 for x in ph2word]  # 0预留给padding
        mel2word = [x + 1 for x in mel2word]  # 0预留给padding
        res['ph_words'] = ph_words  # [T_word]
        res['ph2word'] = ph2word  # [T_ph]
        res['mel2word'] = mel2word  # [T_mel]
        res['dur_word'] = dur_word  # [T_word]

        words = [x for x in res['txt']]
        if words[-1] in PUNCS:
            words = words[:-1]
        words = ['<BOS>'] + words + ['<EOS>']
        word_tokens = word_encoder.encode(" ".join(words))
        res['words'] = words
        res['word_tokens'] = word_tokens
        assert len(words) == len(ph_words), [words, ph_words]

        # words = [x for x in res['txt'].split(" ") if x != '']
        # while len(words) > 0 and is_sil_phoneme(words[0]):
        #     words = words[1:]
        # while len(words) > 0 and is_sil_phoneme(words[-1]):
        #     words = words[:-1]
        # words = ['<BOS>'] + words + ['<EOS>']
        # word_tokens = word_encoder.encode(" ".join(words))
        # res['words'] = words
        # res['word_tokens'] = word_tokens
        # assert len(words) == len(ph_words_nosep), [words, ph_words_nosep]


if __name__ == "__main__":
    set_hparams()
    ZhBinarizer().process()
