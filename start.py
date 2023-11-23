import torch
import wandb

from UrbanSoundPreprocessor import UrbanSoundPreprocessor
from UrbanSoundTrainer import UrbanSoundTrainer


def main():
    print(f"{torch.cuda.is_available()=}")
    print(f"{torch.backends.mps.is_available()=}")
    urbansound_dir = "/Users/davery/urbansound8k"
    n_mels_list = [128]
    n_fft_list = [512]
    chunk_timesteps = [112]
    overwrite_specs = False
    train_only = False
    model_type = "BasicCNN"
    lr = 0.000005
    epochs_per_run = 1
    batch_size = 256
    wandb_run = False

    # if 1, then no mixup applied
    mixup_alpha = 1

    # "hf" for hugging face dataset, "local" for dataset in urbansound_dir
    data_source = "hf"

    print(f"Model: {model_type}")

    for n_mels in n_mels_list:
        for n_fft in n_fft_list:
            for chunk_timestep in chunk_timesteps:
                config = {
                    "model_type": model_type,
                    "n_mels": n_mels,
                    "n_fft": n_fft,
                    "chunk_timestep": chunk_timestep,
                    "train_only": train_only,
                    "lr": lr,
                    "mixup_alpha": mixup_alpha,
                }
                print(config)
                if wandb_run:
                    wandb.init(project="urbansound", config=config)
                print("-" * 50)
                dataset_name = f"n_mels-{n_mels}-n_fft-{n_fft}-chunk-{chunk_timestep}"
                print(dataset_name)
                preprocessor = UrbanSoundPreprocessor(
                    base_dir=urbansound_dir,
                    n_mels=n_mels,
                    dataset_name=dataset_name,
                    n_fft=n_fft,
                    chunk_timesteps=chunk_timestep,
                    fold=None,
                    data_source=data_source,
                )
                if overwrite_specs:
                    preprocessor.run(overwrite=overwrite_specs)
                input_shape = (preprocessor.n_mels, preprocessor.chunk_timesteps)
                print(f"{input_shape=}")
                model_kwargs = {"input_shape": input_shape}
                try:
                    trainer = UrbanSoundTrainer(
                        spec_dir=preprocessor.dest_dir,
                        model_template={
                            "model_type": model_type,
                            "model_kwargs": model_kwargs,
                        },
                        batch_size=batch_size,
                        optim_params={"lr": lr},
                        fold=None,
                        wandb_config=wandb.config if wandb_run else None,
                        mixup_alpha=mixup_alpha,
                    )
                    if train_only:
                        train_loss, train_acc = trainer.run_train_only(
                            epochs=epochs_per_run
                        )
                        print()
                        print(f"{dataset_name=}: {train_loss=} {train_acc=}")
                    else:
                        results = trainer.run(epochs=epochs_per_run, single_fold=1)

                        print()
                        print(
                            f"{dataset_name}: {results['train_loss']=:.5f} {results['train_acc']=:.2f}% {results['val_loss']=:.5f} {results['val_acc']=:.2f}% {results['grouped_acc']=:.2f}%"
                        )
                    if wandb_run:
                        wandb.finish()
                except RuntimeError as e:
                    print(f"error in {dataset_name}: {e}")
                    continue


if __name__ == "__main__":
    main()
