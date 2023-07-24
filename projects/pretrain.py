"""Mathematical expressions pre-training.

Uses mathematical expressions for pre-training followed by training on natural
language.

"""
import torch

from src.config.config import Config
from src.config.config import init_config
from src.dataloader import get_dataloader
from src.modules.model import Transformer
from src.tester import Tester
from src.trainer.trainer import Trainer
from src.utils.tools import set_random_seed
from src.utils.tools import load_checkpoint


def run_training_math(config: Config):
    """Runs training on mathematical expression dataset."""

    dataloader = get_dataloader(config=config)

    # Ensure that input sequence length of math model is equal or larger than
    # maximum size of mathematical expression.
    input_sequence_length = config.model.input_sequence_length
    max_input_length = dataloader.dataset.max_input_length
    assert input_sequence_length >= max_input_length, (
        f"'input_sequence_length' must be larger than 'max_input_length' of" \
        f"'train_dataset'"
    )
    dataloader.dataset.max_input_length = input_sequence_length

    model = Transformer(config=config)
    model.to(config.trainer.device)

    print(config)
    trainer = Trainer(model=model, dataloader=dataloader, config=config)
    trainer.run()

    print("Training finished.")


def run_training_lang(config: Config):
    """Runs training on text dataset."""

    dataloader = get_dataloader(config=config)

    model = build_model(config=config)
    model.to(config.trainer.device)

    print(config)
    trainer = Trainer(model=model, dataloader=dataloader, config=config)
    trainer.run()

    print("Training finished.")


def build_model(config: Config) -> torch.nn.Module:
    """Build model with backbone from pre-trained model."""

    # Get the model.
    model = Transformer(config=config)

    # Load model weights.
    load_checkpoint(
        model=model,
        ckpt_dir=config.dirs.weights,
        model_name=config.load_model.model_name,
    )

    # Extract backbone. TODO
    # Insert into new model. TODO
    
    return model


def run_experiment():
    # Seed random number generator.
    set_random_seed(seed=0)

    # Get configuration file.
    config = init_config(file_path="config.yml")

    # Define pre-training dataset.
    config.dataset.dataset = "boolean"

    # Run pre-training on mathematical expression.
    run_training_math(config=config)

    # Define training dataset.
    config.dataset.dataset = "tinystories"

    # Run training on natural language.
    run_training_lang(config=config)


if __name__ == "__main__":
    run_experiment()
