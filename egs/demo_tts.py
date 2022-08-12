import os
import argparse
# Below provide the TTS pipeline in LJSpeech dataset.

def synthesize(choice, N, text):
    # If you are curious about these choices: https://huggingface.co/spaces/NATSpeech/PortaSpeech/resolve/main/checkpoints/
    exp = ['ps_normal_exp', 'fs2_exp', 'diffspeech']
    steps = ['406000', '160000', '160000']
    infer = ['ps_flow', 'fs2_orig', 'ds']

    # Install dependencies
    if not os.path.exists('PortaSpeech/'):
        print('-----------------------Start installing dependencies-----------------------')
        os.system('git clone https://huggingface.co/spaces/NATSpeech/PortaSpeech.git')
        os.system('cp -r egs/tts/* PortaSpeech/inference/tts/')

    ckpt = f'PortaSpeech/checkpoints/{exp[choice]}/model_ckpt_steps_{steps[choice]}.ckpt'
    if not os.path.exists(ckpt) or os.path.getsize(ckpt) < 1000:
        os.system(
            f'wget https://huggingface.co/spaces/NATSpeech/PortaSpeech/resolve/main/checkpoints/{exp[choice]}/model_ckpt_steps_{steps[choice]}.ckpt')
        os.system(f'mv model_ckpt_steps_406000.ckpt PortaSpeech/checkpoints/{exp[choice]}')

    # TTS
    print(f'-----------------------Start text-to-spectrogram synthesis using {exp[choice]}-----------------------')
    os.system(f" cd PortaSpeech && CUDA_VISIBLE_DEVICES=0 python  inference/tts/{infer[choice]}.py --exp_name {exp[choice]} --hparams='processed_data_dir={text.replace(',', '/')}'")

    # FastDiff
    print('-----------------------Start neural vocoding using FastDiff-----------------------')
    os.system(f"CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config modules/FastDiff/config/FastDiff.yaml --exp_name FastDiff --infer --hparams='test_mel_dir=PortaSpeech/infer_out/,use_wav=False,N={N}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Below provide the TTS pipeline in LJSpeech dataset.")
    parser.add_argument("--N", type=str, default='4', help="denoising steps")
    parser.add_argument("--text", "-o", type=str, help="input text",
                        default="the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing.")
    parser.add_argument("--model", type=int, choices=[0, 1, 2], default=0, help="choice a TTS model.")
    args = parser.parse_args()

    try:
        synthesize(args.model, args.N, args.text)
    except KeyboardInterrupt:
        print('KeyboardInterrupt.')
