import torch
from naturalspeech2_pytorch import (
    EncodecWrapper,
    Model,
    NaturalSpeech2,
    PhonemeEncoder
)
import torchaudio

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    codec = EncodecWrapper()
    model = Model(
        dim=128,
        depth=6,
        dim_prompt=512,
        cond_drop_prob=0.25,  # Dropout for prompt conditioning
        condition_on_prompt=True
    )

    # diffusion model
    diffusion = NaturalSpeech2(
        model=model,
        codec=codec,
        timesteps=1000
    ).to(device)
    
    text = "Hello, I am Abhishek Rajora"

    audio_path = 'audio_test.wav'
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=24000)(waveform)
    raw_audio = waveform.unsqueeze(0).to(device)

    # encoding text (already done in diffusion.sample)
    # text_enc = PhonemeEncoder(text).to(device)

    # Mock raw audio prompt
    prompt = torch.randn(1, 32768).to(device)

    # sampling new audio
    generated_audio = diffusion.sample(
        length=1024,
        text=text,
        prompt=prompt
    ).to(device)

    torchaudio.save('./results/gen_audio.wav', generated_audio.cpu(), sample_rate=24000)