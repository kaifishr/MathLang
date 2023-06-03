import datetime
import os
import pathlib

import torch
from torch.utils.tensorboard import SummaryWriter

from src.config.config import Config
from src.utils.stats import comp_stats_classification
from src.summary.summary import add_graph
from src.summary.summary import add_token_embedding_weights
from src.summary.summary import add_position_embedding_weights
from src.summary.summary import add_linear_weights
from src.summary.summary import add_kernel_weights
from src.summary.summary import add_mask_weights


class Tester:
    """Tester class.

    Attributes:
        model: PyTorch model.
        dataloader: Test dataloader.
        config: Class holding configuration.

    """

    def __init__(
        self, model: torch.nn.Module, 
        dataloader: tuple, 
        config: Config
    ) -> None:
        """Initializes the Tester instance.
        
        Args:
            model:
            dataloader:
            config:
        """
        self.model = model
        self.dataloader = dataloader
        self.config = config

        runs_dir = config.dirs.runs
        dataset = config.dataloader.dataset

        uid = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        tag = config.tag

        log_dir = os.path.join(runs_dir, f"{uid}_{dataset}{f'_{tag}' if tag else ''}")

        self.writer = SummaryWriter(log_dir=log_dir)

        # Save config file to runs.
        file_path = pathlib.Path(self.writer.log_dir) / "config.txt"
        with open(file_path, "w") as file:
            file.write(self.config.__str__())

        self.criterion = torch.nn.CrossEntropyLoss()

        self.running_loss = 0.0
        self.running_accuracy = 0.0
        self.running_counter = 0

    @torch.no_grad()
    def run(self):
        """Main testing logic."""

        config = self.config
        writer = self.writer
        model = self.model
        criterion = self.criterion
        device = config.trainer.device

        num_tests = 10

        for i, (x_data, y_data) in enumerate(self.dataloader):
            # Get the inputs and labels.
            inputs, labels = x_data.to(device), y_data.to(device)

            # Feedforward.
            outputs = model(inputs)

            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)

            # Compute loss.
            loss = criterion(outputs, labels)

            # keeping track of statistics
            self.running_loss += loss.item()
            self.running_accuracy += (
                (torch.argmax(outputs, dim=1) == labels).float().sum()
            )
            self.running_counter += labels.size(0)

            writer.add_scalar(
                "test/loss", 
                self.running_loss / self.running_counter, 
                global_step=i
            )

            writer.add_scalar(
                "test/accuracy", 
                self.running_accuracy / self.running_counter, 
                global_step=i
            )

            if i == num_tests:
                break

        writer.close()