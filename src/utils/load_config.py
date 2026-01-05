"""Utility to load and merge YAML configuration files."""

import yaml


def load_config(config_path):
    """
    Load and merge the agent's YAML configuration with the base configuration.

    Args:
        config_path (str): Path to the agent's YAML configuration file.

    Returns:
        dict: Merged configuration dictionary.
    """
    with open(config_path, "r") as agent_file:
        agent_config = yaml.safe_load(agent_file)

    base_config_path = "configs/base_config.yaml"
    with open(base_config_path, "r") as base_file:
        base_config = yaml.safe_load(base_file)

    # Merge the base config under the 'agent' key
    config = {"agent": base_config, **agent_config}
    return config
