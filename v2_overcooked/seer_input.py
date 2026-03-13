import jax.numpy as jnp


def build_seer_input(state):
    """
    Converts the full game State into a (4, 5, 11) tensor for the Seer CNN.

    Channels:
        0-2  : state.grid (static objects, dynamic objects, extra info)
        3    : agent position (1 where agent is, 0 elsewhere)
        4-7  : agent direction one-hot (UP, DOWN, RIGHT, LEFT) at agent cell
        8-10 : agent inventory one-hot (empty, plate, ingredient) at agent cell

    Recipe and time are NOT included here.
    They get concatenated AFTER the CNN (not as spatial channels).
    """
    height = 4
    width  = 5

    # Channels 0-2: raw grid
    grid_channels = state.grid.astype(jnp.float32)

    # Channel 3: agent position
    agent_x = state.agents.pos.x[0]
    agent_y = state.agents.pos.y[0]

    pos_channel = jnp.zeros((height, width), dtype=jnp.float32)
    pos_channel = pos_channel.at[agent_y, agent_x].set(1.0)
    pos_channel = pos_channel[:, :, None]

    # Channels 4-7: direction one-hot
    agent_dir    = state.agents.dir[0]
    dir_channels = jnp.zeros((height, width, 4), dtype=jnp.float32)
    dir_channels = dir_channels.at[agent_y, agent_x, agent_dir].set(1.0)

    # Channels 8-10: inventory one-hot
    agent_inv    = state.agents.inventory[0]
    inv_channels = jnp.zeros((height, width, 3), dtype=jnp.float32)
    inv_idx      = jnp.clip(agent_inv, 0, 2)
    inv_channels = inv_channels.at[agent_y, agent_x, inv_idx].set(1.0)

    # Stack all channels → (4, 5, 11)
    return jnp.concatenate([
        grid_channels,
        pos_channel,
        dir_channels,
        inv_channels,
    ], axis=-1)
