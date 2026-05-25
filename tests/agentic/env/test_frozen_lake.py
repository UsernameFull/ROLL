from roll.pipeline.agentic.env.frozen_lake import FrozenLakeEnv


def test_frozen_lake_rejects_invalid_action_without_changing_state():
    env = FrozenLakeEnv(size=4, p=1.0, is_slippery=False, map_seed=42, format_penalty=-0.2)
    try:
        obs, info = env.reset(seed=42)

        next_obs, reward, terminated, truncated, step_info = env.step("not tagged")

        assert "P" in obs
        assert "env_instruction" in info
        assert next_obs == obs
        assert reward == -0.2
        assert terminated is False
        assert truncated is False
        assert step_info["metrics"]["action_is_valid"] is False
        assert step_info["metrics"]["action_is_effective"] is False
    finally:
        env.close()
