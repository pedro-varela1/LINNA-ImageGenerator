import json
import os


def load_config(config_path=None):
    """
    Load config.json.  If *config_path* is None, look for config.json in the
    repository root (two levels above this file: utils/ → python_files/).
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.json",
        )
    with open(config_path) as f:
        return json.load(f)
