# SU_Major
This repository contains the modifications of NaturalSpeech2 for Speech Major Exam.

Use the following command to use NaturalSpeech 2 pipeline,
```
pip install naturalspeech2-pytorch
```

Replace the encodec.py file with the modified version which employs masking strategy present in this repository.

For training, run the model on LibriSpeech using,
```
python train.py
```

For inferencing run,
```
python inference.py
```