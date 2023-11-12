# %%
import datetime
import os
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import UrbanSoundPreprocessor
from torch.utils.data import DataLoader, Dataset

print(torch.cuda.is_available())


# %%
class UrbanSoundTrainer:
    def __init__(
        self,
        spec_dir,
        model_class,
        optimizer=optim.Adam,
        optim_params=None,
        loss_function=None,
        fold=1,
        batch_size=32,
    ):
        if optim_params is None:
            self.optim_params = {"lr": 0.0001}
        if loss_function is None:
            self.loss_function = nn.CrossEntropyLoss()
        else:
            self.loss_function = loss_function

        self.model_class = model_class
        self.optimizer = optimizer

        self.spec_dir = spec_dir
        self.batch_size = batch_size
        self.fold = fold

    def prepare_train_val_datasets(self, fold=1):
        val_fold_name = f"fold{fold}"
        spec_dir = Path(self.spec_dir)
        train_folds = [
            d for d in self.spec_dir.iterdir() if d.is_dir() and d.name != val_fold_name
        ]
        val_folds = [spec_dir / val_fold_name]

        train_dataset = SpectrogramCVDataset(spec_fold_dirs=train_folds)
        val_dataset = SpectrogramCVDataset(spec_fold_dirs=val_folds)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
        )
        return train_loader, val_loader

    def prepare_overtrain_dataset(self, fold=None):
        if fold is not None:
            fold_dirs = [self.spec_dir / f"fold{fold}"]
        else:
            fold_dirs = [self.spec_dir / f"fold{i}" for i in range(1, 11)]
        spectrogram_dataset = SpectrogramCVDataset(spec_fold_dirs=fold_dirs)
        spectrogram_dataloader = DataLoader(
            spectrogram_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=True,
            pin_memory=True,
        )
        return spectrogram_dataloader

    def cross_validation_loop(self, num_epochs, single_fold=None):
        print(
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Starting training with cross-validation."
        )
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        if single_fold is None:
            fold_nums = range(1, 11)
        else:
            fold_nums = [single_fold]
        for fold_num in fold_nums:
            print()
            print(f"Val fold: {fold_num}")
            model = self.model_class(input_shape=input_shape).to("cuda")
            optimizer = self.optimizer(model.parameters(), **self.optim_params)
            train_dataloader, val_dataloader = self.prepare_train_val_datasets(
                fold=fold_num
            )
            for epoch in range(num_epochs):
                print(
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Epoch {epoch+1}/{num_epochs}, ",
                    end="\n",
                )
                train_loss, train_acc = self.train(train_dataloader, model, optimizer)
                print(
                    f"\tTrain Loss: {train_loss:.5f}, Train Acc: {train_acc:.2f}%",
                    end="",
                )
                val_loss, val_acc = self.validate(val_dataloader, model)
                print(
                    f"\tVal Loss: {val_loss:.5f}, Val Acc: {val_acc:.2f}%",
                    end="\n",
                )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        print()
        mean_train_loss = statistics.mean(train_losses)
        mean_train_acc = statistics.mean(train_accs)
        mean_val_loss = statistics.mean(val_losses)
        mean_val_acc = statistics.mean(val_accs)
        print(f"Mean training loss: {mean_train_loss:.5f}")
        print(f"Mean training accuracy: {mean_train_acc:.2f}%")
        print(f"Mean validation loss: {mean_val_loss:.5f}")
        print(f"Mean validation accuracy: {mean_val_acc:.2f}%")
        return mean_train_loss, mean_train_acc, mean_val_loss, mean_val_acc

    def training_loop_train_only(self, num_epochs):
        model = self.model_class(input_shape=input_shape).to("cuda")
        optimizer = self.optimizer(model.parameters(), **self.optim_params)
        print(
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Starting training."
        )
        for epoch in range(num_epochs):
            dataloader = self.prepare_overtrain_dataset(self.fold)
            train_loss, train_acc = self.train(dataloader, model, optimizer)
            print(
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Epoch {epoch+1}/{num_epochs}, ",
                end="",
            )
            print(
                f"Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc:.2f}%",
                end="\n",
            )
        return train_loss, train_acc

    def train(self, dataloader, model, optimizer):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to("cuda")
            # print(data.shape)
            data = data.unsqueeze(1)
            data = F.normalize(data, dim=2)

            target = target.to("cuda")
            optimizer.zero_grad()
            output = model(data)
            loss = self.loss_function(output, target)
            epoch_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(output.data, 1)
            epoch_total += target.size(0)
            epoch_correct += (predicted == target).sum().item()

            loss.backward()
            optimizer.step()
        avg_loss = epoch_loss / len(dataloader)
        avg_acc = 100.0 * epoch_correct / epoch_total
        return avg_loss, avg_acc

    def validate(self, dataloader, model):
        model.eval()
        with torch.no_grad():
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.to("cuda")
                data = data.unsqueeze(1)
                data = F.normalize(data, dim=2)

                target = target.to("cuda")
                output = model(data)
                loss = self.loss_function(output, target)
                epoch_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(output.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()
        avg_loss = epoch_loss / len(dataloader)
        avg_acc = 100.0 * epoch_correct / epoch_total
        return avg_loss, avg_acc

    def run_train_only(self, epochs=10):
        return self.training_loop_train_only(epochs)

    def run(self, epochs=10, single_fold=None):
        return self.cross_validation_loop(epochs, single_fold=single_fold)


class SpectrogramDataset(Dataset):
    def __init__(self, spec_dir, transform=None, target_transform=None):
        self.spec_dir = Path(spec_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.spec_paths = list(spec_dir.glob("*.spec"))

    def __len__(self):
        return len(self.spec_paths)

    def __getitem__(self, idx):
        file_path = self.spec_paths[idx]
        file_name = os.path.basename(file_path)
        spec = torch.load(self.spec_paths[idx])
        parts = file_name.split("-")
        label = int(parts[1])
        return spec, label


class SpectrogramCVDataset(Dataset):
    def __init__(self, spec_fold_dirs, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.spec_paths = []
        for fold_dir in spec_fold_dirs:
            self.spec_paths.extend(list(Path(fold_dir).glob("*.spec")))

    def __len__(self):
        return len(self.spec_paths)

    def __getitem__(self, idx):
        file_path = self.spec_paths[idx]
        file_name = Path(file_path).name
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
            dummy_output = self.convolute(dummy_input)
            self.flat_features = int(torch.numel(dummy_output) / dummy_output.shape[0])
        self.fc1 = nn.Linear(self.flat_features, num_classes)

    def convolute(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        return x

    def forward(self, x):
        x = self.convolute(x)
        x = x.view(-1, self.flat_features)
        x = self.fc1(x)
        return x


# %%
class BasicCNN_2(nn.Module):
    def __init__(self, input_shape, num_classes=10):
        super(BasicCNN_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        with torch.no_grad():
            dummy_input = torch.zeros(
                1, 1, input_shape[0], input_shape[1]
            )  # Batch size of 1, 1 channel, input_shape[0] x input_shape[1] image
            dummy_output = self.convolute(dummy_input)
            self.flat_features = int(torch.numel(dummy_output) / dummy_output.shape[0])

        self.fc1 = nn.Linear(self.flat_features, num_classes)

    def convolute(self, x):
        x = self.relu(self.conv1a(self.conv1(x)))
        x = self.pool1(x)
        return x

    def forward(self, x):
        x = self.convolute(x)
        x = x.view(-1, self.flat_features)
        x = self.fc1(x)
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
            dummy_output = self.convolute(dummy_input)
            print(
                f"input_shape: {dummy_input.shape} -> post-convolution: {dummy_output.shape}"
            )
            self.flat_features = int(torch.numel(dummy_output) / dummy_output.shape[0])

        self.fc1 = nn.Linear(self.flat_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def convolute(self, x):
        x = self.relu(self.conv1a(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.conv2a(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.conv3a(self.conv3(x)))
        x = self.pool3(x)
        return x

    def forward(self, x):
        x = self.convolute(x)
        x = x.view(-1, self.flat_features)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x


# %%
if __name__ == "__main__":
    n_mels_list = [75]
    n_fft_list = [512]
    chunk_timesteps = [256, 512, 1024]
    generate_specs = False
    train_only = True

    for n_mels in n_mels_list:
        for n_fft in n_fft_list:
            for chunk_timestep in chunk_timesteps:
                print("-" * 50)
                dataset_name = f"n_mels{n_mels}-n_fft{n_fft}-ct{chunk_timestep}"
                print(dataset_name)
                preprocessor = UrbanSoundPreprocessor(
                    n_mels=n_mels,
                    dataset_name=dataset_name,
                    n_fft=n_fft,
                    chunk_timesteps=chunk_timestep,
                    fold=None,
                )
                if generate_specs:
                    preprocessor.run()
                input_shape = (preprocessor.n_mels, preprocessor.chunk_timesteps)
                print(f"{n_mels=} {input_shape=}")
                try:
                    trainer = UrbanSoundTrainer(
                        spec_dir=preprocessor.dest_dir,
                        input_shape=input_shape,
                        model_class=BasicCNN_3,
                        batch_size=128,
                        fold=None,
                    )
                    if train_only:
                        train_loss, train_acc = trainer.run_train_only(epochs=30)
                        print()
                        print(f"{dataset_name=}: {train_loss=} {train_acc=}")
                    else:
                        train_loss, train_acc, val_loss, val_acc = trainer.run(
                            epochs=15, single_fold=1
                        )
                        print()
                        print(
                            f"{dataset_name}: {train_loss=:.5f} {train_acc=:.2f}% {val_loss=:.5f} {val_acc=:.2f}%"
                        )

                except RuntimeError as e:
                    print(f"error in {dataset_name}: {e}")
                    continue
