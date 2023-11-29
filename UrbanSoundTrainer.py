import datetime
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb
from Mixup import Mixup
from network_factory import network_factory
from SpectrogramDataset import SpectrogramDataset


class UrbanSoundTrainer:
    def __init__(
        self,
        spec_dir,
        model_template,
        wandb_config,
        optimizer=optim.Adam,
        optim_params={"lr": 0.0001},
        loss_function=nn.CrossEntropyLoss(),
        fold=1,
        batch_size=128,
        mixup_alpha=1,
    ):
        self.spec_dir = spec_dir
        self.saved_models_dir = spec_dir.parent / "saved_models"
        self.saved_models_dir.mkdir(parents=True, exist_ok=True)

        self.model_type = model_template["model_type"]
        self.model_kwargs = model_template["model_kwargs"]
        self.optimizer = optimizer
        self.optim_params = optim_params
        self.loss_function = loss_function

        self.wandb_config = wandb_config
        self.fold = fold
        self.batch_size = batch_size

        self.run_timestamp = self.formatted_timestamp(filename=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        if mixup_alpha != 1:
            self.mixup = Mixup(mixup_alpha, device=self.device)
        else:
            self.mixup = None

    def run(self, epochs=10, single_fold=None):
        return self.cross_validation_loop(epochs, single_fold=single_fold)

    def run_train_only(self, epochs=10):
        return self.training_loop_train_only(epochs)

    def cross_validation_loop(self, num_epochs, single_fold=None):
        print(f"{self.formatted_timestamp()}: Starting training with cross-validation.")

        results = {
            "train_losses": [],
            "train_accs": [],
            "val_losses": [],
            "val_accs": [],
            "majority_accs": [],
            "prob_avg_accs": [],
        }

        if single_fold is None:
            fold_nums = range(1, 11)
        else:
            fold_nums = [single_fold]
        for fold_num in fold_nums:
            print()
            print(f"Val fold: {fold_num}")

            model = self.initialize_model()
            optimizer = self.initialize_optimizer(model)
            train_dataloader, val_dataloader = self.prepare_train_val_datasets(
                transforms=model.train_dev_transforms(), fold=fold_num
            )

            for epoch in range(num_epochs):
                self.print_epoch_start(epoch, num_epochs)
                train_loss, train_acc = self.train(model, optimizer, train_dataloader)
                print(
                    f"\tTrain Loss: {train_loss:.5f}, Train Acc: {train_acc:.2f}%",
                    end="",
                )
                val_loss, val_acc, majority_acc, prob_avg_acc = self.validate(
                    model, val_dataloader
                )
                print(f"\tVal Loss: {val_loss:.5f}, Val Acc: {val_acc:.2f}%")
                print(
                    f"\tMajority Acc: {majority_acc:.2f}%, Prob Avg Acc: {prob_avg_acc:.2f}%"
                )
                if self.wandb_config:
                    wandb.log(
                        {
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "majority_acc": majority_acc,
                            "prob_avg_acc": prob_avg_acc,
                        }
                    )
            results["train_losses"].append(train_loss)
            results["train_accs"].append(train_acc)
            results["val_losses"].append(val_loss)
            results["val_accs"].append(val_acc)
            results["majority_accs"].append(majority_acc)
            results["prob_avg_accs"].append(prob_avg_acc)

        self.save_model(model, optimizer)
        print()
        final_results = self.get_result_means(results)

        print(f"Mean training loss: {final_results['train_loss']:.5f}")
        print(f"Mean training accuracy: {final_results['train_acc']:.2f}%")
        print(f"Mean validation loss: {final_results['val_loss']:.5f}")
        print(f"Mean validation accuracy: {final_results['val_acc']:.2f}%")
        print(f"Mean majority vote accuracy: {final_results['majority_acc']:.2f}%")
        print(f"Mean probability avg accuracy: {final_results['prob_avg_acc']:.2f}%")
        return final_results

    def training_loop_train_only(self, num_epochs):
        print(f"{self.formatted_timestamp()}: Starting training.")

        model = self.initialize_model()
        optimizer = self.initialize_optimizer(model)
        dataloader = self.prepare_overtrain_dataset(
            self.fold, transforms=model.train_dev_transforms()
        )

        for epoch in range(num_epochs):
            self.print_epoch_start(epoch, num_epochs)
            train_loss, train_acc = self.train(model, optimizer, dataloader)

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
        self.save_model(model, optimizer)
        return train_loss, train_acc

    def train(self, model, optimizer, dataloader):
        model.train()
        if self.device == "cuda":
            scaler = GradScaler()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (data, target, filenames) in enumerate(tqdm(dataloader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            if data.dim() == 3:
                data = data.unsqueeze(1)

            data = F.normalize(data, dim=2)

            optimizer.zero_grad()
            if self.mixup:
                data, target_a, target_b, lam = self.mixup.mixup_data(data, target)

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

    def validate(self, model, dataloader):
        model.eval()
        with torch.no_grad():
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            all_chunk_probabilities = defaultdict(list)
            for batch_idx, (data, target, filenames) in enumerate(tqdm(dataloader, desc="Validating")):
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
                probabilities = F.softmax(output, dim=1)

                for filename, prob in zip(
                    filenames, probabilities
                ):
                    all_chunk_probabilities[filename].append(prob)

                loss = self.loss_function(output, target)
                epoch_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(output.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()
        avg_loss = epoch_loss / len(dataloader)
        avg_acc = 100.0 * epoch_correct / epoch_total
        majority_acc = self.majority_vote_accuracy(all_chunk_probabilities)
        prob_avg_acc = self.probability_average_accuracy(all_chunk_probabilities)
        return avg_loss, avg_acc, majority_acc, prob_avg_acc

    def forward_pass(self, model, data, target, target_a=None, target_b=None, lam=None):
        output = model(data)
        if self.mixup:
            loss = self.mixup.mixup_loss(
                self.loss_function, output, target_a, target_b, lam
            )
        else:
            loss = self.loss_function(output, target)
        return output, loss

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

    def initialize_model(self):
        return network_factory(model_type=self.model_type, **self.model_kwargs).to(
            self.device
        )

    def initialize_optimizer(self, model):
        return self.optimizer(model.parameters(), **self.optim_params)

    def formatted_timestamp(self, filename=False):
        if filename:
            return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        else:
            return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def print_epoch_start(self, epoch, num_epochs):
        print(
            f"{self.formatted_timestamp()}: Epoch {epoch+1}/{num_epochs}, ",
            end="\n",
        )

    def get_result_means(self, results):
        mean_train_loss = statistics.mean(results["train_losses"])
        mean_train_acc = statistics.mean(results["train_accs"])
        mean_val_loss = statistics.mean(results["val_losses"])
        mean_val_acc = statistics.mean(results["val_accs"])
        mean_majority_vote_acc = statistics.mean(results["majority_accs"])
        mean_prob_avg_acc = statistics.mean(results["prob_avg_accs"])
        return {
            "train_loss": mean_train_loss,
            "train_acc": mean_train_acc,
            "val_loss": mean_val_loss,
            "val_acc": mean_val_acc,
            "majority_acc": mean_majority_vote_acc,
            "prob_avg_acc": mean_prob_avg_acc,
        }

    def save_model(self, model, optimizer):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "wandb_config": self.wandb_config,
            },
            self.saved_models_dir
            / f"model_dict_{self.model_type}_{self.run_timestamp}.pth",
        )

    def majority_vote_accuracy(self, all_chunk_probabilities):
        correct = 0
        total_files = len(all_chunk_probabilities)
        for filename, probs in all_chunk_probabilities.items():
            votes = [torch.argmax(prob).item() for prob in probs]
            vote_count = Counter(votes)
            predicted_label = vote_count.most_common(1)[0][0]
            parts = filename.split("-")
            label = int(parts[1])
            if predicted_label == label:
                correct += 1
        accuracy = (correct / total_files) * 100
        return accuracy

    def probability_average_accuracy(self, all_chunk_probabilities):
        correct = 0
        total_files = len(all_chunk_probabilities)
        for filename, probs in all_chunk_probabilities.items():
            probs_tensor = torch.stack(probs)
            mean_probs = torch.mean(probs_tensor, dim=0)
            parts = filename.split("-")
            label = int(parts[1])
            predicted_label = torch.argmax(mean_probs).item()
            if predicted_label == label:
                correct += 1
        accuracy = (correct / total_files) * 100
        return accuracy
