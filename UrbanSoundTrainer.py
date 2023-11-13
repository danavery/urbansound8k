import datetime
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network_factory import network_factory
from SpectrogramDataset import SpectrogramDataset
from torch.utils.data import DataLoader


class UrbanSoundTrainer:
    def __init__(
        self,
        spec_dir,
        model_type,
        model_kwargs,
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

        self.model_type = model_type
        self.model_kwargs = model_kwargs
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

        train_dataset = SpectrogramDataset(spec_fold_dirs=train_folds)
        val_dataset = SpectrogramDataset(spec_fold_dirs=val_folds)

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
        spectrogram_dataset = SpectrogramDataset(spec_fold_dirs=fold_dirs)
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
            model = network_factory(model_type=self.model_type, **self.model_kwargs).to("cuda")
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
        model = network_factory(model_type=self.model_type, **self.model_kwargs).to("cuda")
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
