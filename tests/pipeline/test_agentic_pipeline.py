import argparse
import json
import os
from dataclasses import asdict

import pytest
from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.agentic.agentic_config import AgenticConfig


DEFAULT_CONFIG_NAME = "agentic_pipeline_config"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PPO Configuration")
    parser.add_argument("--config_name", type=str, default=DEFAULT_CONFIG_NAME, help="Name of the PPO configuration.")
    return parser.parse_args(argv)


def make_ppo_config(config_name=DEFAULT_CONFIG_NAME):
    config_path = "."

    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)
    print(cfg)
    ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))
    return ppo_config


def test_make_ppo_config():
    ppo_config = make_ppo_config()
    print(ppo_config)


@pytest.mark.skipif(
    os.environ.get("RUN_PIPELINE_INTEGRATION") != "1",
    reason="Full pipeline integration run is disabled by default.",
)
def test_ppo_pipeline(config_name=DEFAULT_CONFIG_NAME):
    from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline

    ppo_config = make_ppo_config(config_name)

    init()

    pipeline = AgenticPipeline(pipeline_config=ppo_config)

    pipeline.run()

    output_file = "ppo_pipeline.json"
    with open(output_file, "w") as f:
        json.dump(asdict(pipeline.state), f, ensure_ascii=False)


if __name__ == "__main__":
    cli_args = parse_args()
    test_ppo_pipeline(cli_args.config_name)
