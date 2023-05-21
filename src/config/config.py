"""A simple class that can hold a configuration.
"""
import yaml
from pathlib import Path

import torch


class Config:
    """Configuration class.

    Class creates nested configuration for parameters used
    in different modules during training.
    """

    def __init__(self, d: dict = None) -> None:
        """Initializes config class."""
        self.merge_dict(d)

    @staticmethod
    def _merge_dict(self: object, d: object) -> None:
        if d is not None:
            for key, value in d.items():
                if isinstance(value, dict):
                    if not hasattr(self, key):
                        self.__setattr__(key, Config())
                    self._merge_dict(self=self.__getattribute__(key), d=value)
                else:
                    self.__setattr__(key, value)

    def merge_dict(self, d: dict) -> None:
        self._merge_dict(self, d)

    def __str__(self) -> str:
        """Prints nested config."""
        cfg = []
        self._build_str(self, cfg)
        return "".join(cfg)

    @staticmethod
    def _build_str(self: object, cfg: list, indent: int = 0) -> None:
        """Recursively iterates through all configuration nodes."""
        for key, value in self.__dict__.items():
            indent_ = 4 * indent * " "
            if isinstance(value, Config):
                cfg.append(f"{indent_}{key}\n")
                self._build_str(
                    self=self.__getattribute__(key), cfg=cfg, indent=indent + 1
                )
            else:
                cfg.append(f"{indent_}{key}: {value}\n")


def init_config(file_path: str) -> Config:
    """Initializes configuration class for experiment.

    Args:
        file_path: File to configuration file.

    Returns:
        Config class.
    """
    # Load yaml file as dictionary.
    config = load_config(file_path=file_path)

    # Convert dictionary to configuration class.
    config = Config(d=config)

    Path(config.dirs.runs).mkdir(parents=True, exist_ok=True)
    Path(config.dirs.weights).mkdir(parents=True, exist_ok=True)

    # Check for accelerator
    if config.trainer.device == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config.trainer.device = device
    else:
        config.trainer.device == torch.device("cpu")

    return config


def load_config(file_path: str) -> dict:
    """Loads configuration file.

    Args:
        file_path: Path to yaml file.

    Returns:
        Dictionary holding content of yaml file.
    """
    with open(file_path, "r") as fp:
        try:
            config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)

    return config
