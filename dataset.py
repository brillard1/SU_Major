import os
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram

class LibriSpeechDataset(Dataset):
    def __init__(self, directory, annotations_file='LibriSpeech/test-clean.csv'):
        self.audiopaths = []
        self.transcriptions = []
        self.resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)
        
        annotations = pd.read_csv(annotations_file)

        transcription_dict = {os.path.basename(row['audio-path']): row['transcription'] 
                              for index, row in annotations.iterrows()}

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.flac'):
                    path = os.path.join(root, file)
                    if os.path.basename(path) in transcription_dict:
                        self.audiopaths.append(path)
                        self.transcriptions.append(transcription_dict[os.path.basename(path)])

    def __len__(self):
        return len(self.audiopaths)

    def __getitem__(self, index):
        audio_path = self.audiopaths[index]
        transcription = self.transcriptions[index]
        waveform, _ = torchaudio.load(audio_path)
        mel_spec = self.resample(waveform)
        return mel_spec, transcription