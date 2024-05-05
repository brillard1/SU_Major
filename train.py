from dataset import LibriSpeechDataset
from naturalspeech2_pytorch import Trainer
from naturalspeech2_pytorch import (
    EncodecWrapper,
    Model,
    NaturalSpeech2,
)
import torch

dataset_path = 'LibriSpeech'

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    codec = EncodecWrapper(mask_ratio=0.5) # use the modified one

    model = Model(
        dim = 128,
        depth = 6,
        dim_prompt = 512,
        cond_drop_prob = 0.25,     # dropout prompt conditioning
        condition_on_prompt = True
    )

    # diffusion model
    diffusion = NaturalSpeech2(
        model = model,
        codec = codec,
        timesteps = 1000
    ).to(device)

    dataset = LibriSpeechDataset(dataset_path)

    trainer = Trainer(
        diffusion_model=diffusion,
        dataset = dataset,
        gradient_accumulate_every=2,
        epochs=5,
        results_folder = './results'
    )
    trainer.train()