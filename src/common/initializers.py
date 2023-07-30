"""
Common functions used by all (both train and interact) modules.

Author: Sa1g
Github: https://github.com/sa1g

Date: 2023.07.30

"""


def test_mapping(mapping_function, mapped_agents: dict, env) -> None:
    """
    Checks if mapping between a mapping function and the environment
    is correct.

    Args:
        mapping_function: function that maps agents id (key) to 'mapped_agents' keys.
        mapped_agents: mapped agents dictionary.
        env: RL environment with Gym interface.

    Raise:
        ValueError if mapping is not correct.
    """
    mapping_function = mapping_function()
    obs = env.reset()

    mapped_keys = []
    for key in obs.keys():
        mapped_keys.append(mapping_function(key))

    for key in mapped_keys:
        if key not in mapped_agents.keys():
            raise ValueError(
                f"'mapping_function' doesn't map correctly agents, {key} not mapped correctly"
            )
