import torch
import torch.nn as nn
import numpy as np
import marioai

# ---------------------------------------------------------------------------
# Input feature layout  (57 total)
# ---------------------------------------------------------------------------
# [  0:49]  7×7 vision crop centred on Mario  (rows 8:15, cols 8:15 of 22×22)
# [ 49:50]  can_jump   (float 0/1)
# [ 50:51]  on_ground  (float 0/1)
# [ 51:54]  nearest enemy  (dx, dy, kind_norm)  — zero-filled if absent
# [ 54:57]  2nd nearest enemy (dx, dy, kind_norm) — zero-filled if absent
# ---------------------------------------------------------------------------
INPUT_DIM  = 57
OUTPUT_DIM = 5   # [backward, forward, crouch, jump, speed/bombs]

# Bounding box for the 7×7 crop inside the 22×22 grid.
# Mario is always at the centre of the grid (index 11,11 in 0-based).
_CROP_R0, _CROP_R1 = 8, 15   # rows  [8, 15)  → 7 rows
_CROP_C0, _CROP_C1 = 8, 15   # cols  [8, 15)  → 7 cols

# Maximum absolute coordinate used for enemy normalisation.
# The 22×22 grid covers roughly 22 tiles; tile size ≈ 16 px so ~352 px range.
_ENEMY_NORM = 352.0

# Rough enemy type normalisation: MarioAI kinds range 0–255; divide by 255.
_KIND_NORM  = 255.0


def _build_input(level_scene, can_jump, on_ground, enemies_floats):
    """
    Construct a fixed-size numpy feature vector from raw observation fields.

    Parameters
    ----------
    level_scene    : np.ndarray shape (22,22) or None
    can_jump       : bool or None
    on_ground      : bool or None
    enemies_floats : list of (x, y, kind) triples, or None

    Returns
    -------
    np.ndarray of shape (INPUT_DIM,) dtype float32
    """
    vec = np.zeros(INPUT_DIM, dtype=np.float32)

    # --- 7×7 vision crop ---
    if level_scene is not None:
        crop = level_scene[_CROP_R0:_CROP_R1, _CROP_C0:_CROP_C1]
        vec[0:49] = crop.flatten().astype(np.float32)

    # --- boolean flags ---
    vec[49] = float(can_jump)   if can_jump   is not None else 0.0
    vec[50] = float(on_ground)  if on_ground  is not None else 0.0

    # --- up to 2 nearest enemies ---
    if enemies_floats:
        # enemies_floats is a list of (x, y, kind) tuples.
        # Sort by Euclidean distance from grid centre (11,11) if we had
        # mario_pos; since mario_pos may be None in fast-TCP mode we sort
        # by distance from (0,0) — still gives a consistent ordering.
        def _dist(e):
            return e[0] ** 2 + e[1] ** 2

        sorted_enemies = sorted(enemies_floats, key=_dist)

        for slot, enemy in enumerate(sorted_enemies[:2]):
            base = 51 + slot * 3
            ex, ey, ek = (enemy + (0,))[:3]  # safe unpack even if len<3
            vec[base    ] = float(ex) / _ENEMY_NORM
            vec[base + 1] = float(ey) / _ENEMY_NORM
            vec[base + 2] = float(ek) / _KIND_NORM

    return vec


class MLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid(),   # multi-binary output in [0,1]
        )

    def forward(self, x):
        return self.model(x)


class MLPAgent(marioai.Agent):
    """
    MLP-based Mario agent.

    Input  : 57-dimensional feature vector (7×7 crop + flags + 2 enemies)
    Output : 5-bit action vector [backward, forward, crouch, jump, speed]
    """

    def __init__(self):
        super(MLPAgent, self).__init__()
        self.input_dim  = INPUT_DIM
        self.output_dim = OUTPUT_DIM

        self.mlp = MLP(self.input_dim, self.output_dim)

        # Hard threshold: output > 0.5 → press button
        self.threshold = 0.5

    # ------------------------------------------------------------------
    # marioai.Agent interface
    # ------------------------------------------------------------------

    def sense(self, obs):
        """Unpack observation into agent attributes via parent class."""
        super(MLPAgent, self).sense(obs)

    def act(self):
        if self.level_scene is None:
            return [0, 1, 0, 0, 0]   # safe default: run forward

        # Build the fixed-size input vector
        inputs = _build_input(
            level_scene    = self.level_scene,
            can_jump       = self.can_jump,
            on_ground      = self.on_ground,
            enemies_floats = self.enemies_floats, 
        )

        input_tensor = torch.tensor(inputs, dtype=torch.float32)

        with torch.no_grad():
            output_tensor = self.mlp(input_tensor)

        action_probs = output_tensor.numpy()
        action = (action_probs > self.threshold).astype(int).tolist()
        return action

    # ------------------------------------------------------------------
    # Parameter vector helpers (used by EA / random search)
    # ------------------------------------------------------------------

    def get_param_vector(self):
        """Flatten all MLP parameters into a 1-D numpy array."""
        params = []
        for param in self.mlp.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_param_vector(self, vector):
        """Load a flat parameter vector back into the MLP."""
        offset = 0
        for param in self.mlp.parameters():
            shape = param.shape
            size  = int(np.prod(shape))
            param.data = torch.tensor(
                vector[offset: offset + size].reshape(shape),
                dtype=torch.float32,
            )
            offset += size