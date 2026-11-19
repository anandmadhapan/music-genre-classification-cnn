# Music Genre Classification

College project that classifies 30-second audio clips into 10 genres using convolutional neural networks on mel spectrograms. Written in Python with librosa for audio processing and TensorFlow/Keras for the models.

## Dataset

The project uses the GTZAN dataset: 1,000 WAV clips, 100 each for blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae and rock. It is not stored in this repository (about 1.2 GB). Download it from Kaggle: [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

With the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) installed and an API token set up:

```bash
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip
```

Place the extracted folders so the layout is:

```
Data/genres_original/    audio clips, one folder per genre
Data/images_original/    spectrogram images (not used by the code)
Data/features_*.csv      pre-computed features (not used by the code)
```

`Data/` is in `.gitignore`, so the dataset never gets committed. The original project repository, with the dataset included, is archived at [anandmadhapan/Music-Genre-Classification](https://github.com/anandmadhapan/Music-Genre-Classification).

## Code and models

`code/cnn.py` does everything from the command line:

- splits each clip into slices, augments them, and converts them to mel spectrograms with librosa
- builds a LeNet-style CNN (`lenet`) or a deeper four-block CNN (`cnn_64`)
- trains with Adam and categorical cross-entropy on an 80/10/10 train/validation/test split
- predicts the genre of a single audio file with a saved model

A trained model is saved in `models/cnn64/`. The `png/` folder holds the architecture diagrams, training curves and confusion matrices saved alongside the models, and `Music_Genre_Classification.pdf` / `.pptx` are the project report and slides.

![Training and validation accuracy for the cnn64 model](png/cnn64_acc_epochs.png)

## Running it

```bash
pip install -r requirements.txt
cd code

python cnn.py train ../Data/genres_original -v -s       # train (saves to ../models/model1)
python cnn.py predict ../Data/genres_original -f path/to/clip.wav -m cnn64
```

## Known issues

- In `train()`, the `cnn64` option currently builds the LeNet model as well; it should call `cnn_64()`.
- The number of epochs is fixed at 10 in the script.
- `predict()` calls `librosa.feature.melspectrogram(sample, sr)` with positional arguments, which librosa 0.10+ no longer accepts; hence the `librosa<0.10` pin.
