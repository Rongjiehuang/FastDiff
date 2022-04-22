import os

os.environ["OMP_NUM_THREADS"] = "1"

from data_gen.tts.bin.binarize import binarize
from data_gen.tts.bin.pre_align import pre_align
from data_gen.tts.bin.train_mfa_align import train_mfa_align
from utils.hparams import set_hparams

if __name__ == '__main__':
    set_hparams()
    pre_align()
    train_mfa_align()
    binarize()
