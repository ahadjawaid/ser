import librosa 
from librosa.feature import melspectrogram
from fastai.data.all import *
from tqdm import tqdm

@Transform
def load_audio(path):
    waveform, sample_rate = librosa.load(path)
    return waveform

@Transform
def mel_transform(y):
    return melspectrogram(y=y)

def save_mel(mel, save_path):
    plt.imsave(save_path, arr=mel, origin="lower")

def extension_to_png(path):
    return Path(path).with_suffix(".png")

@Transform
def create_png(path):
    path = Path(path)
    png_path = Path(extension_to_png(path))
    if not png_path.exists():
        mel = mel_pipeline(path)
        save_mel(mel, png_path)

db_transform = Transform(librosa.power_to_db)  
mel_pipeline = Pipeline([load_audio, mel_transform, db_transform])
get_audio_files = FileGetter(extensions=".wav")

def convert_dataset(path, verbose=True):
    files = get_audio_files(path)
    if verbose:
        files = tqdm(files)

    for path in files:
        create_png(path)