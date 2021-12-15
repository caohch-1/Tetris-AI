import itertools

from gym.envs.registration import register as gym_register

env_list: list = []


def register(incomplete_id: str, entry_point: str) -> None:
    """
    Register the custom environments in Gym.

    :param incomplete_id: part of the official environment ID.
    :param entry_point: the Python entry point of the environment.
    """
    assert incomplete_id.startswith(
        "simplifiedtetris-"
    ), 'Env ID should start with "simplifiedtetris-".'
    assert entry_point.startswith(
        "gym_simplifiedtetris.envs:SimplifiedTetris"
    ), 'Entry point should\
            start with "gym_simplifiedtetris.envs:SimplifiedTetris".'
    assert entry_point.endswith("Env"), 'Entry point should end with "Env".'

    grid_dims = [[20, 10], [10, 10], [8, 6], [7, 4]]
    piece_sizes = [4, 3, 2, 1]

    for (height, width), piece_size in list(
        itertools.product(*[grid_dims, piece_sizes])
    ):
        idx = incomplete_id + f"-{height}x{width}-{piece_size}-v0"

        assert idx not in env_list, f"Already registered env id: {idx}"

        gym_register(
            id=idx,
            entry_point=entry_point,
            nondeterministic=True,
            kwargs={
                "grid_dims": (height, width),
                "piece_size": piece_size,
            },
        )
        env_list.append(idx)
