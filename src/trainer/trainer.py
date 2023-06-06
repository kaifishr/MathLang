import datetime
import os
import pathlib

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config.config import Config
from src.summary.summary import add_graph
from src.summary.summary import add_token_embedding_weights
from src.summary.summary import add_position_embedding_weights
from src.summary.summary import add_linear_weights
from src.summary.summary import add_kernel_weights
from src.summary.summary import add_mask_weights


class Trainer:
    """Trainer class.

    Attributes:
        model: PyTorch model.
        dataloader: The training dataloader.
        config: Class holding configuration.

    Typical usage example:
        model = Model()
        dataloader = Dataloader()
        config = Config()
        trainer = Trainer(model, dataloader, config):
        trainer.run()
    """

    def __init__(
        self, 
        model: torch.nn.Module, 
        dataloader: DataLoader, 
        config: Config
    ) -> None:
        """Initializes Trainer."""
        self.model = model
        self.dataloader = dataloader
        self.config = config

        self.num_update_steps = config.trainer.num_update_steps

        runs_dir = config.dirs.runs
        dataset = config.dataloader.dataset

        uid = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        tag = config.tag

        log_dir = os.path.join(runs_dir, f"{uid}_{dataset}{f'_{tag}' if tag else ''}")

        self.writer = SummaryWriter(log_dir=log_dir)

        # Save config file
        file_path = pathlib.Path(self.writer.log_dir) / "config.txt"
        with open(file_path, "w") as file:
            file.write(self.config.__str__())

        # Add graph of model to Tensorboard.
        if config.summary.add_graph:
            add_graph(
                model=model, 
                dataloader=dataloader, 
                writer=self.writer, 
                config=config
            )

        learning_rate = config.trainer.learning_rate
        weight_decay = config.trainer.weight_decay
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Get index of padding token.
        ignore_index = self.dataloader.dataset.char_to_idx[" "]
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )

        self.running_loss = 0.0
        self.running_accuracy = 0.0
        self.running_counter = 0

    def run(self):
        """Main training logic."""

        config = self.config
        writer = self.writer
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        device = config.trainer.device

        update_step = 0

        for x_data, y_data in self.dataloader:
            # Get the inputs and labels.
            inputs, labels = x_data.to(device), y_data.to(device)

            # Zero the parameter gradients.
            optimizer.zero_grad(set_to_none=True)

            # Feedforward.
            outputs = model(inputs)

            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)

            # Compute loss.
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Clip gradients
            if config.trainer.gradient_clipping.is_activated:
                max_norm = config.trainer.gradient_clipping.max_norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # Gradient descent
            optimizer.step()

            # keeping track of statistics
            self.running_loss += loss.item()
            self.running_accuracy += (
                (torch.argmax(outputs, dim=1) == labels).float().sum()
            )
            self.running_counter += labels.size(0)

            self._train_summary(writer=writer, update_step=update_step)
            self._write_summary(writer=writer, model=model, update_step=update_step)

            update_step += 1
            if update_step == self.num_update_steps:
                break

        num_update_steps = config.trainer.num_update_steps
        self._write_summary(model=model, writer=writer, update_step=num_update_steps)
        writer.close()

    def _train_summary(self, writer, update_step: int) -> None:
        """"""
        config = self.config

        if config.summary.save_train_stats.every_n_updates > 0:
            if (update_step + 1) % config.summary.save_train_stats.every_n_updates == 0:

                train_loss = self.running_loss / self.running_counter
                train_accuracy = self.running_accuracy / self.running_counter

                writer.add_scalar(
                    "train/loss", 
                    train_loss, 
                    global_step=update_step
                )

                writer.add_scalar(
                    "train/accuracy", 
                    train_accuracy, 
                    global_step=update_step
                )

                self.running_loss = 0.0
                self.running_accuracy = 0.0
                self.running_counter = 0

                print(f"{update_step:09d} {train_loss:.5f} {train_accuracy:.4f}")

    def _write_summary(self, writer, model, update_step: int) -> None:
        """"""
        config = self.config

        if config.summary.save_model.every_n_updates > 0:
            if update_step % config.summary.save_model.every_n_updates == 0:
                dataset = config.dataloader.dataset
                tag = f"_{config.tag}" if config.tag else ""
                model_name = f"{dataset}{tag}.pth"
                model_path = os.path.join(config.dirs.weights, model_name)
                torch.save(model.state_dict(), model_path)

        if config.summary.add_linear_weights.every_n_updates > 0:
            if update_step % config.summary.add_linear_weights.every_n_updates == 0:
                add_linear_weights(model=model, writer=writer, global_step=update_step)

        if config.summary.add_linear_weights.every_n_updates > 0:
            if update_step % config.summary.add_linear_weights.every_n_updates == 0:
                add_kernel_weights(model=model, writer=writer, global_step=update_step)

        if config.summary.add_token_embeddings.every_n_updates > 0:
            if update_step % config.summary.add_token_embeddings.every_n_updates == 0:
                add_token_embedding_weights(model=model, writer=writer, global_step=update_step)

        if config.summary.add_position_embeddings.every_n_updates > 0:
            if (update_step % config.summary.add_position_embeddings.every_n_updates == 0):
                add_position_embedding_weights(model=model, writer=writer, global_step=update_step)

        if config.summary.add_mask_weights.every_n_updates > 0:
            if (update_step % config.summary.add_mask_weights.every_n_updates == 0):
                add_mask_weights(model=model, writer=writer, global_step=update_step)