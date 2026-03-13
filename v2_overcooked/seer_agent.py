import flax.linen as nn

from seer_input import build_seer_input
from seer_encoder import SeerEncoder
from seer_fsq import SeerFSQ


class SeerAgent(nn.Module):
    """
    Full Seer agent: state encoding + GRU memory + FSQ discretisation.

    Reads the full game State and produces a discrete message for the Cook.
    Returns:
        new_hidden : (128,)  GRU carry for the next timestep
        z_q        : (8,)    quantised message to concatenate into Cook input
        index      : scalar  integer message id, used for logging and TopSim

    Pipeline:
        State -> (4, 5, 11) -> SeerEncoder (CNN + LayerNorm + GRU) -> SeerFSQ -> z_q, index

    (Foerster et al. 2016, Mentzer et al. 2023, ClusterComm 2024)
    """

    @nn.compact
    def __call__(self, hidden, state):
        x = build_seer_input(state)
        new_hidden, gru_output = SeerEncoder()(hidden, x)
        z_q, index = SeerFSQ()(gru_output)
        return new_hidden, z_q, index

    @staticmethod
    def initialize_hidden(rng):
        """Return a zeroed GRU hidden state for the start of an episode."""
        return SeerEncoder.initialize_hidden(rng)
