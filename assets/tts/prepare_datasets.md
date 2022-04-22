# Prepare Datasets

## LJSpeech
### Download dataset

```bash
mkdir -p data/raw/
cd data/raw/
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
bzip2 -d LJSpeech-1.1.tar.bz2
tar -xf LJSpeech-1.1.tar
cd ../../
```

### Download pre-trained vocoder
```
mkdir wavegan_pretrained
```
download `checkpoint-1000000steps.pkl`, `config.yaml`, `stats.h5` from https://drive.google.com/open?id=1XRn3s_wzPF2fdfGshLwuvNHrbgD0hqVS to `wavegan_pretrained/`
    
## VCTK
Download VCTK from https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html and put `VCTK-Corpus` to `data/raw/VCTK-Corpus`


## THCHS30

Download THCHS from https://www.openslr.org/18/ to `data/raw/thchs30`

## AIShell-3

Download AIShell-3 from http://www.aishelltech.com/aishell_3 to `data/raw/AISHELL-3`

## LibriTTS

Download LibriTTS (`train-clean-100.tar.gz`, `train-clean-360.tar.gz`) from http://www.openslr.org/60/ to `data/raw/LibriTTS`

## DidiSpeech
Download DidiSpeech from https://outreach.didichuxing.com/app-vue/DatasetProjectDetail?id=1021 to `data/raw/didispeech`
