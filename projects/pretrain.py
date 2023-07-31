"""Mathematical expressions pre-training.

Uses mathematical expressions for pre-training for subsequent training on a
natural language dataset:

1) Pre-training with mathematical expressions.
2) Use backbone from pre-trained model for new NLP model.
    - Token embedding, positional embedding and classification head are new.
3) Traing new model on natural language data.

"""
import torch

from src.config.config import Config
from src.config.config import init_config
from src.dataloader import get_dataloader
from src.modules.model import Transformer
from src.trainer.trainer import Trainer
from src.utils.tools import set_random_seed


def run_training_math(config: Config) -> torch.nn.Module:
    """Runs training on mathematical expression dataset."""

    dataloader = get_dataloader(config=config)

    # Ensure that input sequence length of math model is equal or larger than
    # maximum size of mathematical expression.
    input_sequence_length = config.model.input_sequence_length
    max_input_length = dataloader.dataset.max_input_length
    print(f"{max_input_length = }")
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

    return model


def run_training_lang(math_model: torch.nn.Module, config: Config):
    """Runs training on text dataset."""

    dataloader = get_dataloader(config=config)
    
    model = Transformer(config=config)
    model.to(config.trainer.device)

    # Replace model's backbone with pre-trained weights.
    model.transformer_blocks = math_model.transformer_blocks

    print(config)
    trainer = Trainer(model=model, dataloader=dataloader, config=config)
    trainer.run()

    print("Training finished.")


def run_experiment():
    # Seed random number generator.
    set_random_seed(seed=0)

    # Get configuration file.
    config = init_config(file_path="config.yml")

    # Define pre-training dataset.
    config.trainer.num_update_steps = 20000
    config.dataset.dataset = "arithmetic"
    config.load_model.model_name = config.dataset.dataset

    # Run pre-training on mathematical expression.
    math_model = run_training_math(config=config)

    # Define training dataset.
    config.trainer.num_update_steps = 20000
    config.dataset.dataset = "tinystories"
    config.model.output_sequence_length = 1

    # Run training on natural language.
    run_training_lang(math_model=math_model, config=config)


if __name__ == "__main__":
    run_experiment()
