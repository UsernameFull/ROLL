import asyncio
from dataclasses import dataclass, field

import pytest
import ray

from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager
from roll.pipeline.agentic.agentic_config import EnvMonitorConfig


class NoOpGroupFilter:
    def __init__(self, *args, **kwargs):
        pass

    def filter(self, group_id: int, episode_id: int, group: list[DataProto]):
        return False


@dataclass
class MockAgenticConfig:
    rollout_batch_size: int
    async_generation_ratio: int
    env_monitor: EnvMonitorConfig = field(default_factory=lambda: EnvMonitorConfig(enable=False))


class MockEnvManagerConfig:
    def __init__(self, rollout_batch_size: int, group_size: int = 2, env_groups: int = 2):
        self.world_size = 1
        self.env_groups = env_groups
        self.group_size = group_size
        self.group_size_redundancy = 0
        self.group_filter_cls = "tests.agentic.rollout.test_rollout_scheduler.NoOpGroupFilter"

        train_env_num = env_groups * group_size
        self.max_traj_per_env = (rollout_batch_size + train_env_num - 1) // train_env_num
        self.max_env_num_per_worker = train_env_num
        self.env_configs = {
            0: {
                env_id: {"group_id": env_id // group_size}
                for env_id in range(train_env_num)
            }
        }


async def _put_one_group(output_queue, group_id: int, group_size: int, step: int):
    episode_ids = []
    for env_offset in range(group_size):
        env_id = group_id * group_size + env_offset
        episode_id = await output_queue.get_episode_id.remote(group_id, env_id)
        episode_ids.append(episode_id)

    assert len(set(episode_ids)) == 1

    for env_offset, episode_id in enumerate(episode_ids):
        rollout = DataProto(meta_info={"group_id": group_id, "env_offset": env_offset, "step": step})
        await output_queue.put.remote(group_id, episode_id, step, rollout)


async def _run_group_queue_manager_smoke():
    rollout_batch_size = 4
    config = MockAgenticConfig(rollout_batch_size=rollout_batch_size, async_generation_ratio=0)
    env_manager_config = MockEnvManagerConfig(rollout_batch_size=rollout_batch_size)
    env_num = env_manager_config.world_size * env_manager_config.max_env_num_per_worker

    output_queue = GroupQueueManager.options(max_concurrency=env_num + 1).remote(
        config,
        env_manager_config,
        "train",
    )

    try:
        for step in range(2):
            await output_queue.advance_step.remote(step)
            await asyncio.gather(
                *[
                    _put_one_group(output_queue, group_id, env_manager_config.group_size, step)
                    for group_id in range(env_manager_config.env_groups)
                ]
            )

            batch = await output_queue.get_batch.remote(batch_size=rollout_batch_size, current_step=step)
            assert len(batch) == rollout_batch_size
            assert {rollout.meta_info["group_id"] for rollout in batch} == {0, 1}
            assert all(rollout.meta_info["step"] == step for rollout in batch)
    finally:
        await output_queue.shutdown.remote()


@pytest.mark.skip_on_npu
def test_group_queue_manager_cpu_smoke():
    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)

    try:
        asyncio.run(_run_group_queue_manager_smoke())
    finally:
        if started_ray:
            ray.shutdown()
