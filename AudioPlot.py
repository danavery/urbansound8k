import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import Audio


class AudioPlot:
    def plot_saved_spec(self, path):
        spec = torch.load(path)
        self.plot_spec(spec)

    def plot_sample_spec(self):
        self.plot_saved_spec(
            "/home/davery/ml/urbansound8k/processed/fold1/203356-3-0-3.wav-0.spec"
        )

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

    def frame_timings(self, spec):
        num_frames = spec.shape[-1]
        time_per_frame = self.hop_length / self.sample_rate
        time_values = (torch.arange(0, num_frames) * time_per_frame).numpy()
        return num_frames, time_per_frame, time_values
