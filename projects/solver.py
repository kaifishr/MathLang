"""
Neural network-based solver for arithmetic, algebraic, and boolean expressions.
"""
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from src.config.config import Config
from src.config.config import init_config
from src.modules.model import MLPMixer
from src.modules.model import ConvMixer
from src.modules.model import ConvModel
from src.modules.model import Transformer
from src.utils.tools import load_checkpoint
from src.utils.tools import set_random_seed
from src.utils.tools import count_model_parameters
from src.dataloader import get_dataloader


class Solver:
    """Solver class for mathematical expressions.

    This class loads a pre-trained model that can be used as a solver for 
    arithmetic, algebraic, and boolean expressions.

    Attributes:
        model: An autoregressive model.
        dataset: Dataset model has been trained on.
        config: Configuration.
        valid_characters: Legal characters.
        device: Selected hardware accelerator.
        max_input_length: Maximum number of tokens of input sequence.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        config: Config,
    ) -> None:
        """Initializes Solver."""
        self.model = model
        self.dataset = dataset
        self.config = config

        self.char_to_idx = self.dataset.char_to_idx
        self.idx_to_char = self.dataset.idx_to_char

        self.valid_tokens = list(self.char_to_idx)
        self.device = self.config.trainer.device
        self.max_input_length = self.dataset.max_input_length

    def _is_valid_sequence(self, sequence: str) -> bool:
        """Checks if input prompt contains any illegal tokens."""
        for token in sequence:
            if token not in self.valid_tokens:
                print(f"\nToken '{token}' was not part of the training data.")
                return False
        return True

    def _add_padding(self, sequence: str) -> str:
        """Pads sequence to correct size."""
        return sequence.ljust(self.max_input_length, " ")
    
    def _solve(self, expression: str) -> str:
        """Solves expression."""

        # Encode input characters as integer using lookup table from dataloader.
        data = [self.char_to_idx[token] for token in expression]

        # Create input tensor from encoded characters.
        x = torch.tensor(data=data, dtype=torch.long)[None, ...].to(self.device)

        # Feed sequence into model.
        logits = self.model(x)

        # Convert logits to probabilities (not really necessary here)
        probabilities = F.softmax(input=logits, dim=-1).squeeze()

        # Select the most likely tokens.
        indices = torch.argmax(probabilities, dim=-1)

        output = "".join([self.idx_to_char[int(index)] for index in indices])

        return output

    def test(self):
        """Tests model with some simple prompts."""

        expressions = [
            "(1+2)-3+5",
            "2+(1+2)-4+1",
            "(4-(3-4)+(4+9))",
        ]

        for expression in expressions:
            print(f"\n>>>\n{expressions}\n")
            if self._is_valid_sequence(sequence=expression):
                expression = self._add_padding(sequence=expression)
                print(f"{expression = }")
                output = self._solve(expression=expression)
                print(f"\n>>> {output}\n")
                print(f"\n>>> true {eval(expression)}\n")

    def run(self):
        """Runs solver."""
        is_running = True

        print("\nPlease enter an expression.\n")

        while is_running:
            print("[User]")
            expression = input()

            if expression.startswith("!"):
                command = expression[1:]
                if command == "exit":
                    is_running = False
                else:
                    print(f"Command '{command}' not recognized.")
                continue
            elif expression == "":
                continue

            # Feed expression to model
            if is_running and self._is_valid_sequence(sequence=expression):
                expression = self._add_padding(sequence=expression)
                output = self._solve(expression=expression)
                print(f"\n['Solver']\n{output}\n")


if __name__ == "__main__":
    # Get configuration file.
    config = init_config(file_path="config.yml")

    # Seed random number generator.
    set_random_seed(seed=config.random_seed)

    # Get dataset.
    dataloader = get_dataloader(config=config)
    dataset = dataloader.dataset

    # Get the model.
    model_type = config.model.type
    if model_type == "convmixer":
        model = ConvMixer(config=config)
    elif model_type == "cnn":
        model = ConvModel(config=config)
    elif model_type == "mlpmixer":
        model = MLPMixer(config=config)
    elif model_type == "transformer":
        model = Transformer(config=config)
    else:
        raise NotImplementedError(f"Model type {model_type} not available.")

    count_model_parameters(model=model)
    model.to(config.trainer.device)
    model = torch.jit.script(model)

    ckpt_dir = config.dirs.weights
    model_name = config.load_model.model_name
    load_checkpoint(model=model, ckpt_dir=ckpt_dir, model_name=model_name)
    model.to(config.trainer.device)
    model.eval()

    solver = Solver(model=model, dataset=dataset, config=config)
    solver.test()
    solver.run()
