import datetime
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.distributions.beta import Beta
from torch.utils.data import DataLoader

import wandb
from network_factory import network_factory
from SpectrogramDataset import SpectrogramDataset


class UrbanSoundTrainer:
    def __init__(
        self,
        spec_dir,
        model_template,
        wandb_config,
        optimizer=optim.Adam,
        optim_params=None,
        loss_function=None,
        fold=1,
        batch_size=128,
        mixup_alpha=1,
    ):
        if optim_params is None:
            self.optim_params = {"lr": 0.0001}
        if loss_function is None:
            self.loss_function = nn.CrossEntropyLoss()
        else:
            self.loss_function = loss_function

        self.model_type = model_template["model_type"]
        self.model_kwargs = model_template["model_kwargs"]
        self.optimizer = optimizer
        self.optim_params = optim_params

        self.spec_dir = spec_dir
        self.saved_models_dir = spec_dir.parent / "saved_models"
        self.saved_models_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.fold = fold
        self.wandb_config = wandb_config
        self.mixup_alpha = mixup_alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.run_timestamp = self.formatted_timestamp(filename=True)

    def prepare_train_val_datasets(self, fold=1, transforms=None):
        if transforms is None:
            transforms = {"train": None, "val": None}
        val_fold_name = f"fold{fold}"
        spec_dir = Path(self.spec_dir)
        train_folds = [
            d for d in self.spec_dir.iterdir() if d.is_dir() and d.name != val_fold_name
        ]
        val_folds = [spec_dir / val_fold_name]

        train_dataset = SpectrogramDataset(
            spec_fold_dirs=train_folds, transform=transforms["train"]
        )
        val_dataset = SpectrogramDataset(
            spec_fold_dirs=val_folds, transform=transforms["val"]
        )

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

    def prepare_overtrain_dataset(self, fold=None, transforms=None):
        if transforms is None:
            transforms = {"train": None, "val": None}
        if fold is not None:
            fold_dirs = [self.spec_dir / f"fold{fold}"]
        else:
            fold_dirs = [self.spec_dir / f"fold{i}" for i in range(1, 11)]
        spectrogram_dataset = SpectrogramDataset(
            spec_fold_dirs=fold_dirs, transform=transforms["train"]
        )
        spectrogram_dataloader = DataLoader(
            spectrogram_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=True,
            pin_memory=True,
        )
        return spectrogram_dataloader

    def cross_validation_loop(self, num_epochs, single_fold=None):
        print(f"{self.formatted_timestamp()}: Starting training with cross-validation.")
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        grouped_accs = []
        if single_fold is None:
            fold_nums = range(1, 11)
        else:
            fold_nums = [single_fold]
        for fold_num in fold_nums:
            print()
            print(f"Val fold: {fold_num}")
            model = network_factory(model_type=self.model_type, **self.model_kwargs).to(
                self.device
            )
            optimizer = self.optimizer(model.parameters(), **self.optim_params)
            train_dataloader, val_dataloader = self.prepare_train_val_datasets(
                transforms=model.train_dev_transforms(), fold=fold_num
            )
            for epoch in range(num_epochs):
                print(
                    f"{self.formatted_timestamp()}: Epoch {epoch+1}/{num_epochs}, ",
                    end="\n",
                )
                train_loss, train_acc = self.train(train_dataloader, model, optimizer)
                print(
                    f"\tTrain Loss: {train_loss:.5f}, Train Acc: {train_acc:.2f}%",
                    end="",
                )
                val_loss, val_acc, grouped_acc = self.validate(val_dataloader, model)
                print(
                    f"\tVal Loss: {val_loss:.5f}, Val Acc: {val_acc:.2f}%, Grouped Acc: {grouped_acc:.2f}%",
                    end="\n",
                )
                if self.wandb_config:
                    wandb.log(
                        {
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "grouped_acc": grouped_acc,
                        }
                    )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            grouped_accs.append(grouped_acc)
        print()
        mean_train_loss = statistics.mean(train_losses)
        mean_train_acc = statistics.mean(train_accs)
        mean_val_loss = statistics.mean(val_losses)
        mean_val_acc = statistics.mean(val_accs)
        mean_grouped_acc = statistics.mean(grouped_accs)
        print(f"Mean training loss: {mean_train_loss:.5f}")
        print(f"Mean training accuracy: {mean_train_acc:.2f}%")
        print(f"Mean validation loss: {mean_val_loss:.5f}")
        print(f"Mean validation accuracy: {mean_val_acc:.2f}%")
        print(f"Mean grouped accuracy: {mean_grouped_acc:.2f}%")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "wandb_config": self.wandb_config,
            },
            self.saved_models_dir
            / f"model_dict_{self.model_type}_{self.run_timestamp}.pth",
        )
        return (
            mean_train_loss,
            mean_train_acc,
            mean_val_loss,
            mean_val_acc,
            mean_grouped_acc,
        )

    def formatted_timestamp(self, filename=False):
        if filename:
            return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        else:
            return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def training_loop_train_only(self, num_epochs):
        model = network_factory(model_type=self.model_type, **self.model_kwargs).to(
            self.device
        )
        optimizer = self.optimizer(model.parameters(), **self.optim_params)
        print(f"{self.formatted_timestamp()}: Starting training.")
        for epoch in range(num_epochs):
            dataloader = self.prepare_overtrain_dataset(
                self.fold, transforms=model.train_dev_transforms()
            )
            train_loss, train_acc = self.train(dataloader, model, optimizer)
            print(
                f"{self.formatted_timestamp()}: Epoch {epoch+1}/{num_epochs}, ",
                end="",
            )
            print(
                f"Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc:.2f}%",
                end="\n",
            )
            if self.wandb_config:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                    }
                )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "wandb_config": self.wandb_config,
            },
            self.saved_models_dir / f"model_dict_{self.run_timestamp}.pth",
        )
        return train_loss, train_acc

    def sample_beta_distribution(self, alpha, size=1, device="cuda"):
        distribution = Beta(
            torch.full((size,), alpha, device=device),
            torch.full((size,), alpha, device=device),
        )
        return distribution.sample()

    def mixup_data(self, x, y, alpha=1.0):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if alpha > 0:
            lam = self.sample_beta_distribution(alpha, device=self.device)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def forward_pass(self, model, data, target, target_a=None, target_b=None, lam=None):
        output = model(data)
        if self.mixup_alpha != 1:
            loss = self.mixup_criterion(
                self.loss_function, output, target_a, target_b, lam
            )
        else:
            loss = self.loss_function(output, target)
        return output, loss

    def train(self, dataloader, model, optimizer):
        model.train()
        if self.device == "cuda":
            scaler = GradScaler()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (data, target, filenames) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            if data.dim() == 3:
                data = data.unsqueeze(1)

            data = F.normalize(data, dim=2)

            optimizer.zero_grad()
            if self.mixup_alpha != 1:
                data, target_a, target_b, lam = self.mixup_data(
                    data, target, alpha=self.mixup_alpha
                )

            if self.device == "cuda":
                with autocast():
                    output, loss = self.forward_pass(
                        model, data, target, target_a, target_b, lam
                    )
            else:
                output, loss = self.forward_pass(model, data, target)

            epoch_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(output.data, 1)
            epoch_total += target.size(0)
            epoch_correct += (predicted == target).sum().item()

            if self.device == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        avg_loss = epoch_loss / len(dataloader)
        avg_acc = 100.0 * epoch_correct / epoch_total
        return avg_loss, avg_acc

    def majority_vote(self, all_chunk_predictions):
        correct = []
        for filename, votes in all_chunk_predictions.items():
            vote_count = Counter(votes)
            winner = vote_count.most_common(1)[0][0]
            parts = filename.split("-")
            label = int(parts[1])
            correct.append(winner == label)
        return (sum(correct) / len(correct)) * 100

    def validate(self, dataloader, model):
        model.eval()
        with torch.no_grad():
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            all_chunk_predictions = defaultdict(list)
            for batch_idx, (data, target, filenames) in enumerate(dataloader):
                data = data.to(self.device)
                if data.dim() == 3:
                    data = data.unsqueeze(1)
                data = F.normalize(data, dim=2)

                target = target.to(self.device)
                if self.device == "cuda":
                    with autocast():
                        output = model(data)
                else:
                    output = model(data)

                _, predicted = torch.max(output.data, 1)
                for filename, prediction in zip(filenames, predicted):
                    all_chunk_predictions[filename].append(prediction.item())

                loss = self.loss_function(output, target)
                epoch_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(output.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()
        avg_loss = epoch_loss / len(dataloader)
        avg_acc = 100.0 * epoch_correct / epoch_total
        grouped_accuracy = self.majority_vote(all_chunk_predictions)

        return avg_loss, avg_acc, grouped_accuracy

    def run_train_only(self, epochs=10):
        return self.training_loop_train_only(epochs)

    def run(self, epochs=10, single_fold=None):
        return self.cross_validation_loop(epochs, single_fold=single_fold)
