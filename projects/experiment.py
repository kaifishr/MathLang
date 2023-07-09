"""Main script to run experiments.
"""
from src.config.config import init_config
from src.dataloader import get_dataloader
from src.modules.model import Transformer
from src.tester import Tester
from src.utils.tools import set_random_seed
from src.utils.tools import load_checkpoint


def run_experiment(num_iter: int):
    # Get configuration file.
    config = init_config(file_path="config.yml")

    # Seed random number generator.
    set_random_seed(seed=0)

    # Get dataloader.
    dataloader = get_dataloader(config=config)

    # Get the model.
    model = Transformer(config=config, num_iter=num_iter)

    # Load model weights.
    load_checkpoint(
        model=model,
        ckpt_dir=config.dirs.weights,
        model_name=config.load_model.model_name,
    )

    # Move model to GPU if possible.
    model.to(config.trainer.device)
    model.eval()

    # Run experiment.
    tester = Tester(model=model, dataloader=dataloader, config=config)
    tester.run()


if __name__ == "__main__":
    num_iters = [1, 2, 4, 8, 16, 32, 64, 128]
    for num_iter in num_iters:
        run_experiment(num_iter=num_iter)
