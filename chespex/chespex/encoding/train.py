""" Training of the molecule autoencoder. """

from typing import Optional
import numpy as np
import torch
from .autoencoder import Autoencoder
from .loss import graph_loss


class MoleculeTrainer:
    """
    Trainer class for the molecule autoencoder.
    """

    def __init__(
        self,
        model: Autoencoder,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda",
        data_transform: Optional[callable] = None,
        decoder_weight_decay: float = 0.001,
        **optimizer_kwargs,
    ):
        """
        Initializes the MoleculeTrainer.
        :param model: Autoencoder model to train
        :param optimizer: Torch optimizer to use for training, if None, Adam is used
            with the specified decoder weight decay and optimizer kwargs.
        :param device: Device to use for training
        :param decoder_weight_decay: Weight decay for the decoder
        :param optimizer_kwargs: Additional keyword arguments for the Adam optimizer
        """
        self.model = model
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": model.encoder.parameters()},
                    {
                        "params": model.decoder.parameters(),
                        "weight_decay": decoder_weight_decay,
                    },
                ],
                **optimizer_kwargs,
            )
        self.device = device
        self.data_transform = data_transform
        self.batch_losses = []
        self.batch_accuracies = []
        self.batch_sizes = []
        self.epoch_losses = []
        self.epoch_accuracies = []

    def _print_or_write_progress(
        self,
        print_progress,
        write_progress_to_file,
        number_of_last_steps,
        batch_count=None,
    ):
        """
        Print or write the progress of the training.
        :param print_progress: Whether to print the progress to the stdout.
        :param write_progress_to_file: Filename to write the progress to. If None,
            no file is written.
        :param number_of_last_steps: Number of last steps to average for the progress.
        :param batch_count: The current batch count if relevant.
        :return: The average loss and accuracies over the last steps.
        """
        batch_counts = np.array(self.batch_sizes[-number_of_last_steps:])
        batch_losses = np.array(self.batch_losses[-number_of_last_steps:])
        batch_accuracies = np.array(self.batch_accuracies[-number_of_last_steps:])
        loss_mean = np.sum(batch_losses * batch_counts[:, 0]) / np.sum(
            batch_losses[:, 0]
        )
        accuracies_mean = np.sum(batch_accuracies * batch_counts) / np.sum(
            batch_counts, axis=0
        )
        progress_string = (
            f"Epoch {len(self.epoch_losses) + 1}, " + f"Batch {batch_count}, "
            if batch_count is not None
            else ""
            + f"Loss: {loss_mean:.4f}, Accuracy: "
            + f"edges {accuracies_mean[0]:.3f}, class {accuracies_mean[1]:.3f}, "
            + f"size {accuracies_mean[2]:.3f}, charge {accuracies_mean[3]:.3f}"
        )
        if print_progress:
            print(progress_string)
        if write_progress_to_file is not None:
            with open(write_progress_to_file, "a", encoding="utf-8") as log_file:
                log_file.write(progress_string + "\n")
        return loss_mean, accuracies_mean

    def train(
        self,
        epochs: int,
        dataloader: torch.utils.data.DataLoader,
        print_progress: bool = False,
        write_progress_to_file: Optional[str] = None,
        progress_print_interval: int = 500,
        save_model: Optional[str] = None,
        iteration_wrapper: Optional[callable] = None,
    ):
        """
        Train the model for a number of epochs.
        :param epochs: Number of epochs to train for.
        :param dataloader: DataLoader to use for training.
        :param print_progress: Whether to print the progress of training.
        :param write_progress_to_file: File to write the progress to. If None, no file is written.
        :param progress_print_interval: Interval to print the progress to the stdout and/or file.
            The printed progress is the average loss and accuracy over the last interval.
        :param save_model: Whether to save the model if the training is finished or interrupted.
            If a string is provided, the model is saved to that file. If None, the model is
            not saved.
        :param iteration_wrapper: Function to wrap the training epoch iteration, e.g., tqdm.
        """
        if iteration_wrapper is None:

            def iteration_wrapper(x):
                return x

        try:
            for _ in iteration_wrapper(range(epochs)):
                self.model.train()
                batch_count = 0
                for batch in dataloader:
                    self.optimizer.zero_grad(set_to_none=True)
                    # Prepare batch
                    batch = batch.to(self.device)
                    if self.data_transform is not None:
                        transform_result = self.data_transform(batch)
                        if transform_result is not None:
                            batch = transform_result
                    # Forward pass
                    nodes, adjacency_matrix = self.model(batch)
                    loss, accuracy = graph_loss(
                        batch, nodes, adjacency_matrix, return_accuracy=True
                    )
                    # Backward pass and optimization
                    loss.backward()
                    self.optimizer.step()
                    # Store batch results
                    self.batch_losses.append(loss.item())
                    self.batch_accuracies.append(np.nan_to_num(accuracy))
                    self.batch_sizes.append((1 - np.isnan(accuracy)) * len(batch))
                    batch_count += 1
                    # Print progress
                    if batch_count % progress_print_interval == 0:
                        self._print_or_write_progress(
                            print_progress,
                            write_progress_to_file,
                            progress_print_interval,
                            batch_count,
                        )
                # Store epoch results and reset batch results
                epoch_loss, epoch_accuracies = self._print_or_write_progress(
                    print_progress, write_progress_to_file, progress_print_interval
                )
                self.epoch_losses.append(epoch_loss)
                self.epoch_accuracies.append(epoch_accuracies)
                self.batch_losses = []
                self.batch_accuracies = []
                self.batch_sizes = []
        except KeyboardInterrupt:
            pass
        finally:
            if save_model is not None:
                torch.save(self.model, save_model)
