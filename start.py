import torch
import wandb

from UrbanSoundPreprocessor import UrbanSoundPreprocessor
from UrbanSoundTrainer import UrbanSoundTrainer


def main():
    print(f"{torch.cuda.is_available()=}")
    print(f"{torch.backends.mps.is_available()=}")

    # "hf" for hugging face dataset, "local" for dataset in source_dir
    data_source = "hf"
    source_dir = "/Users/davery/urbansound8k"
    overwrite_specs = False
    train_only = False
    epochs_per_run = 1
    wandb_run = False

    model_type = "BasicCNN"
    n_mels_list = [128]
    n_fft_list = [1024]
    chunk_timesteps = [112]
    lr = 0.000005
    mixup_alpha = 1  # if 1, then no mixup applied
    batch_size = 256

    print(f"Model: {model_type}")

    for n_mels in n_mels_list:
        for n_fft in n_fft_list:
            for chunk_timestep in chunk_timesteps:
                config = {
                    "model_type": model_type,
                    "n_mels": n_mels,
                    "n_fft": n_fft,
                    "chunk_timestep": chunk_timestep,
                    "lr": lr,
                    "mixup_alpha": mixup_alpha,
                    "batch_size": batch_size,
                }
                print(config)
                print("-" * 50)
                if wandb_run:
                    wandb.init(project="urbansound", config=config)
                preprocessor, dataset_name = create_preprocessor(
                    config, data_source, source_dir, overwrite_specs
                )
                try:
                    trainer = create_trainer(
                        preprocessor, config, batch_size, mixup_alpha, wandb_run
                    )
                    run_training(trainer, epochs_per_run, dataset_name, train_only)
                    if wandb_run:
                        wandb.finish()
                except RuntimeError as e:
                    print(f"error in {dataset_name}: {e}")
                    continue


def create_preprocessor(config, data_source, source_dir, overwrite_specs):
    dataset_name = f"n_mels-{config['n_mels']}-n_fft-{config['n_fft']}-chunk-{config['chunk_timestep']}"
    preprocessor = UrbanSoundPreprocessor(
        base_dir=source_dir,
        n_mels=config["n_mels"],
        dataset_name=dataset_name,
        n_fft=config["n_fft"],
        chunk_timesteps=config["chunk_timestep"],
        fold=None,
        data_source=data_source,
    )
    preprocessor.run(overwrite=overwrite_specs)
    return preprocessor, dataset_name


def create_trainer(preprocessor, config, batch_size, mixup_alpha, wandb_run):
    input_shape = (preprocessor.n_mels, preprocessor.chunk_timesteps)
    print(f"{input_shape=}")
    trainer = UrbanSoundTrainer(
        spec_dir=preprocessor.dest_dir,
        model_template={
            "model_type": config["model_type"],
            "model_kwargs": {"input_shape": input_shape},
        },
        batch_size=batch_size,
        optim_params={"lr": config["lr"]},
        fold=None,
        wandb_config=wandb.config if wandb_run else None,
        mixup_alpha=mixup_alpha,
    )
    return trainer


def run_training(trainer, epochs_per_run, dataset_name, train_only):
    if train_only:
        train_loss, train_acc = trainer.run_train_only(epochs=epochs_per_run)
        print(f"\n{dataset_name=}: {train_loss=} {train_acc=}")
    else:
        results = trainer.run(epochs=epochs_per_run, single_fold=1)
        print(f"\n{dataset_name}:")
        print(f"\t{results['train_loss']=:.5f} {results['train_acc']=:.2f}%", end="")
        print(f"\t{results['val_loss']=:.5f} {results['val_acc']=:.2f}%")
        print(f"\t{results['majority_acc']=:.2f}% {results['prob_avg_acc']=:.2f}%")


if __name__ == "__main__":
    main()
