import yaml
import argparse

# Load the config file
with open('config.yml') as f:
    config = yaml.safe_load(f)

config_ns = argparse.Namespace(**config)
