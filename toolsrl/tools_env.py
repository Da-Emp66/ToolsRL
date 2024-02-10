import pygame
import yaml


class ToolsBase():
    def __init__(
            self,
            configuration_filename: str = "./configurations/tools_env_config.yaml"
        ):
        pass

    def __post_init__(self):
        with open(self.configuration_file) as cfg:
            overrides = yaml.safe_load(cfg)
            cfg.close()
        for key in overrides:
            if not self.__dict__.get(key, True):
                self.__setattr__(key, overrides.get(key))

    