# %%

import csv
import datetime
import os
import shutil
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from IPython.display import Audio
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, Resample
from tqdm.notebook import tqdm

print(torch.cuda.is_available())


# %%
class UrbanSoundPreprocessor:
    def __init__(
        self,
        sample_rate=22050,
        n_fft=256,
        hop_factor=0.5,
        n_mels=100,
        chunk_timesteps=1000,
        dataset_name="default",
        fold=1,
    ):
        self.base_dir = Path("/home/davery/ml/urbansound8k")
        self.source_dir = self.base_dir
        self.dest_dir = self.base_dir / "processed" / str(dataset_name)
        self.index_path = self.base_dir / "UrbanSound8K.csv"
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = int(self.n_fft * hop_factor)
        self.n_mels = n_mels
        self.chunk_timesteps = chunk_timesteps
        self.fold = fold

    def load_index(self):
        index = {}
        with open(self.index_path, encoding="UTF-8") as index_file:
            csv_reader = csv.DictReader(index_file)
            index = [row for row in csv_reader]
        return index

    def preprocess(self, filepath):
        audio, file_sr = torchaudio.load(filepath)

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if file_sr != self.sample_rate:
            resampler = Resample(file_sr, self.sample_rate)
            audio = resampler(audio)

        num_samples = audio.shape[-1]
        total_duration = num_samples / self.sample_rate

        return audio, self.sample_rate, num_samples, total_duration

    def make_mel_spectrogram(self, audio):
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

    def frame_timings(self, spec):
        num_frames = spec.shape[-1]
        time_per_frame = self.hop_length / self.sample_rate
        time_values = (torch.arange(0, num_frames) * time_per_frame).numpy()
        return num_frames, time_per_frame, time_values

    def preprocess_all_folds(self, index):
        for record in tqdm(index, total=len(index)):
            fold_dir = Path(f"fold{record['fold']}")
            file_name = record["slice_file_name"]
            source = self.source_dir / fold_dir / file_name
            dest_fold_dir = self.dest_dir / fold_dir
            Path.mkdir(dest_fold_dir, exist_ok=True, parents=True)
            dest_file = dest_fold_dir / f"{file_name}.spec"

            audio, _ = self.preprocess(source)
            mel_spec_db = self.make_mel_spectrogram(audio)

            torch.save(mel_spec_db, dest_file)

    def plot_saved_spec(self, path):
        spec = torch.load(path)
        self.plot_spec(spec)

    def plot_spec(self, spec):
        _, _, time_values = self.frame_timings(spec)
        fig, axs = plt.subplots(1, 1, figsize=(4, 4))

        axs.set_xticks(
            np.arange(0, len(time_values), step=int(len(time_values) / 5)),
            np.round(time_values[:: int(len(time_values) / 5)], 2),
        )

        axs.imshow(spec.numpy(), origin="lower")
        plt.show()

    def plot_audio(self, audio):
        mel_spec_db = self.make_mel_spectrogram(audio)

        _, _, time_values = self.frame_timings(mel_spec_db)

        fig, axs = plt.subplots(2, 1, figsize=(8, 4))
        plt.style.use("dark_background")

        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Amplitude")
        axs[0].plot(audio.t().numpy())

        axs[1].set_xticks(
            np.arange(0, len(time_values), step=int(len(time_values) / 5)),
            np.round(time_values[:: int(len(time_values) / 5)], 2),
        )
        axs[1].imshow(mel_spec_db.numpy())
        plt.show()
        Audio(audio, rate=self.sample_rate)

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

    def test_split_spectrogram(self):
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
        split = self.split_spectrogram(minispec, 2)
        print(f"{split=}")
        print(f"{split.shape=}")

    def create_split_mel_specs(self, index):
        shutil.rmtree(self.dest_dir, ignore_errors=True)
        count = 0
        for record in tqdm(index, total=len(index)):
            if self.fold and record["fold"] != str(self.fold):
                continue
            fold_dir_name = f"fold{record['fold']}"
            file_name = record["slice_file_name"]
            source = self.source_dir / fold_dir_name / file_name
            fold_dir = self.dest_dir / fold_dir_name
            Path.mkdir(fold_dir, exist_ok=True, parents=True)

            audio, sr, num_samples, total_duration = self.preprocess(source)
            mel_spec_db = self.make_mel_spectrogram(audio)
            chunks = self.split_spectrogram(mel_spec_db, self.chunk_timesteps)

            for i in range(len(chunks)):
                dest_file = fold_dir / f"{file_name}-{i}.spec"
                torch.save(chunks[i], dest_file)
                count += 1
        print(f"{count} chunk specs saved")

    def show_sample(self):
        audio, sr, _, _ = self.preprocess(
            "/home/davery/ml/urbansound8k/fold1/203356-3-0-3.wav"
        )
        mel_spec_db = self.make_mel_spectrogram(audio)
        print(f"{mel_spec_db.shape=}")
        split_spec = self.split_spectrogram(mel_spec_db, self.chunk_timesteps)
        print(f"{split_spec.shape=}")
        print("mel_spec_db:")
        self.plot_spec(mel_spec_db)
        for i in range(len(split_spec)):
            self.plot_spec(split_spec[i])
        Audio("/home/davery/ml/urbansound8k/fold1/203356-3-0-3.wav")

    def plot_sample_spec(self):
        self.plot_saved_spec(
            "/home/davery/ml/urbansound8k/processed/fold1/203356-3-0-3.wav-0.spec"
        )

    def run(self):
        index = self.load_index()
        self.create_split_mel_specs(index)
        # self.plot_sample_spec()


# %%
class UrbanSoundTrainer:
    def __init__(
        self,
        spec_dir,
        model_class,
        input_shape,
        optimizer=optim.Adam,
        optim_params=None,
        loss_function=None,
        fold=1,
        batch_size=32,
    ):
        if optim_params is None:
            optim_params = {"lr": 0.0001}
        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()

        self.model = model_class(input_shape=input_shape).to("cuda")
        self.optimizer = optimizer(self.model.parameters(), **optim_params)
        self.loss_function = loss_function
        self.spec_dir = spec_dir
        self.batch_size = batch_size
        self.dataloader = self.prepare_dataset(fold=fold)

    def prepare_dataset(self, fold=None):
        if fold is not None:
            spectrogram_dataset = SpectrogramDataset(
                spec_dir=self.spec_dir / f"fold{fold}"
            )
        else:
            all_folds = [
                SpectrogramDataset(spec_dir=self.spec_dir / f"fold{i}")
                for i in range(1, 11)
            ]
            spectrogram_dataset = ConcatDataset(all_folds)
        spectrogram_dataloader = DataLoader(
            spectrogram_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=True,
            pin_memory=True,
        )
        return spectrogram_dataloader

    def training_loop(self, num_epochs):
        print(
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Starting training."
        )
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train()
            print(
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Epoch {epoch+1}/{num_epochs}, ",
                end="",
            )
            print(
                f"Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc:.2f}%",
                end="\n",
            )
        return train_loss, train_acc

    def train(self):
        self.model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch_idx, (data, target) in enumerate(self.dataloader):
            data = data.to("cuda")
            # print(data.shape)
            data = data.unsqueeze(1)
            data = F.normalize(data, dim=2)

            target = target.to("cuda")
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
            epoch_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(output.data, 1)
            epoch_total += target.size(0)
            epoch_correct += (predicted == target).sum().item()

            loss.backward()
            self.optimizer.step()
        avg_loss = epoch_loss / len(self.dataloader)
        avg_acc = 100.0 * epoch_correct / epoch_total
        return avg_loss, avg_acc

    def run(self, epochs=10):
        return self.training_loop(epochs)


class SpectrogramDataset(Dataset):
    def __init__(self, spec_dir, transform=None, target_transform=None):
        self.spec_dir = spec_dir
        self.transform = transform
        self.target_transform = target_transform
        self.spec_paths = [
            os.path.join(self.spec_dir, file)
            for file in glob("*.spec", root_dir=self.spec_dir)
        ]

    def __len__(self):
        return len(self.spec_paths)

    def __getitem__(self, idx):
        file_path = self.spec_paths[idx]
        file_name = os.path.basename(file_path)
        spec = torch.load(self.spec_paths[idx])
        parts = file_name.split("-")
        label = int(parts[1])
        return spec, label


# %%
class BasicCNN(nn.Module):
    def __init__(self, input_shape, num_classes=10):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        with torch.no_grad():
            dummy_input = torch.zeros(
                1, 1, input_shape[0], input_shape[1]
            )  # Batch size of 1, 1 channel, input_shape[0] x input_shape[1] image
            dummy_output = self.pre_flatten(dummy_input)
            print(f"{dummy_input.shape=} {dummy_output.shape=}")
            self.flat_features = int(torch.numel(dummy_output) / dummy_output.shape[0])
        self.fc1 = nn.Linear(self.flat_features, num_classes)

    def pre_flatten(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        return x

    def forward(self, x):
        x = self.pre_flatten(x)
        x = x.view(-1, self.flat_features)
        x = self.fc1(x)
        return x


# %%
class BasicCNN_2(nn.Module):
    def __init__(self, num_classes=10):
        super(BasicCNN_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        with torch.no_grad():
            dummy_input = torch.zeros(
                1, 1, 100, 500
            )  # Batch size of 1, 1 channel, 100x500 image
            dummy_output = self.pool1(self.conv1(dummy_input))
            self.flat_features = int(torch.numel(dummy_output) / dummy_output.shape[0])

        self.fc1 = nn.Linear(self.flat_features, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1a(self.conv1(x)))
        x = self.pool1(x)
        x = x.view(-1, self.flat_features)
        x = self.relu(self.fc1(x))
        return x


# %%
class BasicCNN_3(nn.Module):
    def __init__(self, input_shape, num_classes=10):
        super(BasicCNN_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=1)
        self.conv1a = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1)
        self.conv2a = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1)
        self.conv3a = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        with torch.no_grad():
            dummy_input = torch.zeros(
                1, 1, input_shape[0], input_shape[1]
            )  # Batch size of 1, 1 channel, input_shape[0] x input_shape[1] image
            dummy_output = self.pre_flatten(dummy_input)
            print(f"{dummy_input.shape=} {dummy_output.shape=}")
            self.flat_features = int(torch.numel(dummy_output) / dummy_output.shape[0])

        self.fc1 = nn.Linear(self.flat_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def pre_flatten(self, x):
        x = self.relu(self.conv1a(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.conv2a(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.conv3a(self.conv3(x)))
        x = self.pool3(x)
        return x

    def forward(self, x):
        x = self.pre_flatten(x)
        x = x.view(-1, self.flat_features)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x


# %%
if __name__ == "__main__":
    for n_mels in [10, 25, 30, 50, 75, 100, 150]:
        preprocessor = UrbanSoundPreprocessor(n_mels=n_mels, dataset_name=n_mels)
        preprocessor.run()
        input_shape = (preprocessor.n_mels, preprocessor.chunk_timesteps)
        print(f"{n_mels=} {input_shape=}")
        try:
            trainer = UrbanSoundTrainer(
                spec_dir=preprocessor.dest_dir,
                input_shape=input_shape,
                model_class=BasicCNN,
                batch_size=8,
            )
        except RuntimeError as e:
            print(f"error in n_mels {n_mels}: {e}")
            continue
        train_loss, train_acc = trainer.run(epochs=15)
        print(f"{n_mels=}: {train_loss=} {train_acc=}")
