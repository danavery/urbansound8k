import csv
import shutil
import os
from pathlib import Path

from datasets import load_dataset
import torch
import torchaudio
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, Resample
from tqdm import tqdm


class UrbanSoundPreprocessor:
    def __init__(
        self,
        base_dir="/home/davery/ml/urbansound8k",
        sample_rate=22050,
        n_fft=256,
        hop_factor=0.5,
        n_mels=100,
        chunk_timesteps=1000,
        dataset_name="default",
        fold=1,
        data_source="local",
    ):
        self.base_dir = Path(base_dir)
        self.source_dir = self.base_dir
        self.dest_dir = self.base_dir / "processed" / str(dataset_name)
        self.index_path = self.base_dir / "UrbanSound8K.csv"
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = int(self.n_fft * hop_factor)
        self.n_mels = n_mels
        self.chunk_timesteps = chunk_timesteps
        self.fold = fold
        self.data_source = data_source
        if self.data_source == "hf":
            self.dataset = load_dataset("danavery/urbansound8k")["train"]

    def run(self, overwrite):
        self.create_split_mel_specs(overwrite)

    def create_split_mel_specs(self, overwrite):
        if os.path.exists(self.dest_dir) and not overwrite:
            return
        shutil.rmtree(self.dest_dir, ignore_errors=True)
        count = 0

        data_iterable = self.get_data_iterable()
        for record in data_iterable:
            if self.fold and record["fold"] != str(self.fold):
                continue

            audio, file_sr = self.load_audio_data(record)
            chunks = self.process_audio(audio, file_sr)

            self.save_chunks(count, record, chunks)
            count += len(chunks)

        print(f"{count} chunk specs saved")

    def load_audio_data(self, record):
        if self.data_source == "local":
            filepath = (
                self.source_dir / f"fold{record['fold']}" / record["slice_file_name"]
            )
            audio, sr = torchaudio.load(filepath)
        elif self.data_source == "hf":
            audio_data = record["audio"]
            audio, sr = audio_data["array"], audio_data["sampling_rate"]
            audio = torch.tensor(audio)
            audio = audio.float()
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)  # Add a channel dimension
        return audio, sr

    def get_data_iterable(self):
        if self.data_source == "local":
            data_index = self.load_index()
            data_iterable = tqdm(data_index, total=len(data_index))
        elif self.data_source == "hf":
            data_iterable = tqdm(self.dataset, total=len(self.dataset))
        return data_iterable

    def load_index(self):
        if self.data_source == "local":
            index = {}
            with open(self.index_path, encoding="UTF-8") as index_file:
                csv_reader = csv.DictReader(index_file)
                index = [row for row in csv_reader]
            return index
        else:
            return None

    def preprocess(self, audio, file_sr):
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if file_sr != self.sample_rate:
            resampler = Resample(file_sr, self.sample_rate)
            audio = resampler(audio)
            audio = audio.float()

        num_samples = audio.shape[-1]
        total_duration = num_samples / self.sample_rate

        return audio, num_samples, total_duration

    def make_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        spec_transformer = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        mel_spec = spec_transformer(audio).squeeze(0)

        amplitude_to_db_transformer = AmplitudeToDB()
        mel_spec_db = amplitude_to_db_transformer(mel_spec)

        return mel_spec_db

    def split_spectrogram(self, spec: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """
        Splits a spectrogram tensor into equal-sized chunks along the time axis.

        This function divides a 2D spectrogram tensor into smaller chunks of a specified size. If the spectrogram
        cannot be evenly divided, the remaining part is zero-padded at the end to form a complete chunk. The output
        is a 3D tensor where the first dimension corresponds to the chunk index.

        Parameters:
        spec (torch.Tensor): A 2D tensor representing the spectrogram with shape (frequency_bins, time_steps).
        chunk_size (int): The desired number of time steps in each chunk.

        Returns:
        torch.Tensor: A 3D tensor with shape (num_chunks, frequency_bins, chunk_size), where num_chunks is the
                    number of total chunks calculated based on the spectrogram size and chunk_size.

        """
        # Calculate number of chunks needed without padding
        new_spec = spec.clone()
        num_chunks = new_spec.shape[1] // chunk_size

        # calculate the size of the remainder
        remainder = new_spec.shape[1] % chunk_size
        if remainder != 0:
            # if there is a remainder, we need to pad the spec
            padding_size = chunk_size - remainder
            padding = torch.zeros(
                (new_spec.shape[0], padding_size),
                dtype=new_spec.dtype,
                device=new_spec.device,
            )
            new_spec = torch.cat([new_spec, padding], dim=1)
            num_chunks += 1
        # Use unfold to split the tensor along the time axis
        unfolded = new_spec.unfold(dimension=1, size=chunk_size, step=chunk_size)

        # unfolded has shape (frequency_bins, num_chunks, chunk_size)
        # We need to transpose it to get (num_chunks, frequency_bins, chunk_size)
        chunks = unfolded.transpose(0, 1)
        return chunks.contiguous()

    def save_chunks(self, count, record, chunks):
        fold_dir_name = f"fold{record['fold']}"
        fold_dir = self.dest_dir / fold_dir_name
        Path.mkdir(fold_dir, exist_ok=True, parents=True)
        file_name = record["slice_file_name"]

        for i in range(len(chunks)):
            dest_file = fold_dir / f"{file_name}-{i}.spec"
            torch.save(chunks[i], dest_file)

    def process_audio(self, audio, file_sr):
        audio, num_samples, total_duration = self.preprocess(audio, file_sr)
        mel_spec_db = self.make_mel_spectrogram(audio)
        chunks = self.split_spectrogram(mel_spec_db, self.chunk_timesteps)
        return chunks


if __name__ == "__main__":
    usp = UrbanSoundPreprocessor()

    def test_split_spectrogram():
        minispec = torch.tensor(
            [
                [5, 6, 7, 8, 9],
                [5, 6, 7, 8, 9],
                [5, 6, 7, 8, 9],
                [5, 6, 7, 8, 9],
                [5, 6, 7, 8, 9],
            ]
        )
        print(f"{minispec=}")
        print(f"{minispec.shape=}")
        split = usp.split_spectrogram(minispec, 2)
        print(f"{split=}")
        print(f"{split.shape=}")

    test_split_spectrogram()
