"""Main script to run trainings."""
from src.config.config import init_config
from src.dataloader import get_dataloader
from src.modules.model import MLPMixer
from src.modules.model import ConvMixer
from src.modules.model import ConvModel
from src.modules.model import Transformer
from src.trainer.trainer import Trainer
from src.utils.tools import set_random_seed
from src.utils.tools import count_model_parameters


def run_experiment():
    # Get configuration file.
    config = init_config(file_path="config.yml")

    # Seed random number generator.
    set_random_seed(seed=config.random_seed)

    # Get dataloader.
    dataloader = get_dataloader(config=config)

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

    print(config)
    trainer = Trainer(model=model, dataloader=dataloader, config=config)
    trainer.run()

    print("Training finished.")


if __name__ == "__main__":
    run_experiment()