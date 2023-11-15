import torch
import wandb

from UrbanSoundPreprocessor import UrbanSoundPreprocessor
from UrbanSoundTrainer import UrbanSoundTrainer

print(torch.cuda.is_available())


if __name__ == "__main__":
    n_mels_list = [50, 100]
    n_fft_list = [256, 512, 1024]
    chunk_timesteps = [256, 512, 1024]
    generate_specs = True
    train_only = False
    model_type = "BasicCNN_3"
    lr = 0.00001
    epochs_per_run = 5
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
                }
                wandb.init(project="urbansound", config=config)
                print("-" * 50)
                dataset_name = f"n_mels-{n_mels}-n_fft-{n_fft}-chunk-{chunk_timestep}"
                print(dataset_name)
                preprocessor = UrbanSoundPreprocessor(
                    base_dir="/home/davery/ml/urbansound8k",
                    n_mels=n_mels,
                    dataset_name=dataset_name,
                    n_fft=n_fft,
                    chunk_timesteps=chunk_timestep,
                    fold=None,
                )
                if generate_specs:
                    preprocessor.run()
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
                        batch_size=32,
                        optim_params={"lr": lr},
                        fold=None,
                        wandb_config=wandb.config,
                    )
                    if train_only:
                        train_loss, train_acc = trainer.run_train_only(
                            epochs=epochs_per_run
                        )
                        print()
                        print(f"{dataset_name=}: {train_loss=} {train_acc=}")
                    else:
                        train_loss, train_acc, val_loss, val_acc = trainer.run(
                            epochs=epochs_per_run, single_fold=1
                        )

                        print()
                        print(
                            f"{dataset_name}: {train_loss=:.5f} {train_acc=:.2f}% {val_loss=:.5f} {val_acc=:.2f}%"
                        )
                    wandb.finish()
                except RuntimeError as e:
                    print(f"error in {dataset_name}: {e}")
                    continue
