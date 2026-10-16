"""CONTRACT
Importing this package registers the Gymnasium id "ChemVialWeighing-v0"
(entry point chem_rl_sim.env:VialWeighingEnv; kwargs pass through gym.make):

    import gymnasium as gym
    import chem_rl_sim  # noqa: F401
    env = gym.make("ChemVialWeighing-v0", normalized_actions=True)

The robot (gantry UR7e) comes from the urgantry_sim git submodule at
third_party/urgantry_sim, fitted with Robotiq 2F-85 grippers by default
(chem_rl_sim/robot.py) or its own Wuji/Sharpa hands with hand="wuji"|"sharpa".
`git submodule update --remote` pulls the latest MaxLab robot model. No max_episode_steps here: the env truncates itself,
same as urgantry_sim's SimGantryUR7e-v0.
"""
import gymnasium as gym

from .env import VialWeighingEnv

ENV_ID = "ChemVialWeighing-v0"

if ENV_ID not in gym.envs.registry:
    gym.register(id=ENV_ID, entry_point="chem_rl_sim.env:VialWeighingEnv")

__all__ = ["VialWeighingEnv", "ENV_ID"]
