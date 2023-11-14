import torch
from UrbanSoundPreprocessor import UrbanSoundPreprocessor
from UrbanSoundTrainer import UrbanSoundTrainer

print(torch.cuda.is_available())

# %%
if __name__ == "__main__":
    n_mels_list = [75]
    n_fft_list = [512]
    chunk_timesteps = [256, 512, 1024]
    generate_specs = True
    train_only = False
    model_type = "BasicCNN"

    for n_mels in n_mels_list:
        for n_fft in n_fft_list:
            for chunk_timestep in chunk_timesteps:
                print("-" * 50)
                dataset_name = f"n_mels-{n_mels}-n_fft-{n_fft}-chunk-{chunk_timestep}"
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
                print(f"{input_shape=}")
                model_kwargs = {"input_shape": input_shape}
                print(f"{n_mels=} {model_kwargs}")
                try:
                    trainer = UrbanSoundTrainer(
                        spec_dir=preprocessor.dest_dir,
                        model_type=model_type,
                        model_kwargs=model_kwargs,
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
