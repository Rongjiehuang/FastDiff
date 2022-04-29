from data_gen.tts.vocoder_pre_align import VocoderPreAlign


class LJPreAlign(VocoderPreAlign):
    def meta_data(self):
        for l in open(f'{self.raw_data_dir}/metadata.csv').readlines():
            item_name, _, txt = l.strip().split("|")
            wav_fn = f"{self.raw_data_dir}/wavs/{item_name}.wav"
            yield item_name, wav_fn, txt, 'SPK1'


if __name__ == "__main__":
    LJPreAlign().process()
