import re
import jieba
from pypinyin import pinyin, Style
from data_gen.tts.data_gen_utils import PUNCS
from data_gen.tts.txt_processors.base_text_processor import BaseTxtProcessor
from utils.text_norm import NSWNormalizer

ALL_SHENMU = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j',
              'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']


class TxtProcessor(BaseTxtProcessor):
    table = {ord(f): ord(t) for f, t in zip(
        u'：，。！？【】（）％＃＠＆１２３４５６７８９０',
        u':,.!?[]()%#@&1234567890')}

    @staticmethod
    def sp_phonemes():
        return ['|', '#']

    @staticmethod
    def preprocess_text(text):
        text = text.translate(TxtProcessor.table)
        text = NSWNormalizer(text).normalize(remove_punc=False).lower()
        text = re.sub("[\'\"()]+", "", text)
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ A-Za-z\u4e00-\u9fff{PUNCS}]", "", text)
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r"", text)
        text = re.sub(rf"[A-Za-z]+", r"$", text)
        return text

    @classmethod
    def pinyin_with_en(cls, txt, style):
        x = pinyin(txt, style)
        x = [t[0] for t in x]
        x_ = []
        for t in x:
            if '$' not in t:
                x_.append(t)
            else:
                x_ += list(t)
        x_ = [t if t != '$' else 'ENG' for t in x_]
        return x_

    @classmethod
    def process(cls, txt, pre_align_args):
        txt = cls.preprocess_text(txt)

        # https://blog.csdn.net/zhoulei124/article/details/89055403
        shengmu = cls.pinyin_with_en(txt, style=Style.INITIALS)
        yunmu = cls.pinyin_with_en(txt, style=
        Style.FINALS_TONE3 if pre_align_args['use_tone'] else Style.FINALS)
        assert len(shengmu) == len(yunmu)
        ph_list = []
        for a, b in zip(shengmu, yunmu):
            if a == b:
                ph_list += [a]
            else:
                ph_list += [a + "%" + b]
        seg_list = '#'.join(jieba.cut(txt))
        assert len(ph_list) == len([s for s in seg_list if s != '#']), (ph_list, seg_list)

        # 加入词边界'#'
        ph_list_ = []
        seg_idx = 0
        for p in ph_list:
            if seg_list[seg_idx] == '#':
                ph_list_.append('#')
                seg_idx += 1
            elif len(ph_list_) > 0:
                ph_list_.append("|")
            seg_idx += 1
            finished = False
            if not finished:
                ph_list_ += [x for x in p.split("%") if x != '']

        ph_list = ph_list_

        # 去除静音符号周围的词边界标记 [..., '#', ',', '#', ...]
        sil_phonemes = list(PUNCS) + TxtProcessor.sp_phonemes()
        ph_list_ = []
        for i in range(0, len(ph_list), 1):
            if ph_list[i] != '#' or (ph_list[i - 1] not in sil_phonemes and ph_list[i + 1] not in sil_phonemes):
                ph_list_.append(ph_list[i])
        ph_list = ph_list_
        return ph_list, txt


if __name__ == '__main__':
    t = 'simon演唱过后，simon还进行了simon精彩的文艺演出simon.'
    phs, txt = TxtProcessor.process(t, {'use_tone': True})
    print(phs, txt)
