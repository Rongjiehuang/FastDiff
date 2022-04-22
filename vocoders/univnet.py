import glob
import re
import torch
import utils
from modules.univnet.generator import Generator
from utils.hparams import hparams, set_hparams
from vocoders.base_vocoder import register_vocoder
from vocoders.pwg import PWG


def load_model(config_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dict = torch.load(checkpoint_path, map_location="cpu")
    config = set_hparams(config_path, global_hparams=False)
    state = ckpt_dict["state_dict"]["model_gen"]
    model = Generator(config)
    model.load_state_dict(state, strict=True)
    model.remove_weight_norm()
    model.eval()
    model.to(device)
    print(f"| Loaded model parameters from {checkpoint_path}.")
    print(f"| UnivNet device: {device}.")
    return model, config, device


total_time = 0


@register_vocoder
class UnivNet(PWG):
    def __init__(self):
        base_dir = hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        ckpt = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
        lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x)[0]))[-1]
        print('| load UnivNet: ', ckpt)
        self.model, self.config, self.device = load_model(config_path=config_path, checkpoint_path=ckpt)

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).transpose(2, 1).to(device)
            noise = torch.randn(1, self.config['gen']['noise_dim'], c.shape[2]).to(device)
            with utils.Timer('UnivNet', enable=hparams['profile_infer']):
                y = self.model(c, noise).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out
