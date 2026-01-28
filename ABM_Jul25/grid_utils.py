# grid_utils.py
import numpy as np
from . import params as P

def laplacian(field: np.ndarray) -> np.ndarray:
    """
    Compute the 2D Laplacian of 'field' using a 5‐point stencil,
    with zero‐flux (Neumann) boundary conditions (∂C/∂n = 0 at edges).
    This implements the ∇2C term in Eq. S1.
    """
    # Initialize an empty array for the Laplacian
    lap = np.zeros_like(field)
    
    # Interior points (i = 1..W-2, j = 1..H-2)
    lap[1:-1, 1:-1] = (
        field[0:-2, 1:-1]   # up
      + field[2:  , 1:-1]   # down
      + field[1:-1, 0:-2]   # left
      + field[1:-1, 2:  ]   # right
      - 4 * field[1:-1, 1:-1]
    )
    
    # Edges: replicate neighbor value to enforce ∂C/∂n = 0
    # Top row (i=0, j=1..H-2)
    lap[0, 1:-1] = (
        field[0, 1:-1]      # itself (reflect)
      + field[1, 1:-1]      # downward neighbor
      + field[0, 0:-2]      # left neighbor
      + field[0, 2:  ]      # right neighbor
      - 4 * field[0, 1:-1]
    )
    # Bottom row (i=W-1, j=1..H-2)
    lap[-1, 1:-1] = (
        field[-1, 1:-1]
      + field[-2, 1:-1]
      + field[-1, 0:-2]
      + field[-1, 2:  ]
      - 4 * field[-1, 1:-1]
    )
    # Left column (i=1..W-2, j=0)
    lap[1:-1, 0] = (
        field[1:-1, 0]
      + field[0:-2, 0]
      + field[2:  , 0]
      + field[1:-1, 1]
      - 4 * field[1:-1, 0]
    )
    # Right column (i=1..W-2, j=H-1)
    lap[1:-1, -1] = (
        field[1:-1, -1]
      + field[0:-2, -1]
      + field[2:  , -1]
      + field[1:-1, -2]
      - 4 * field[1:-1, -1]
    )
    # Corners
    # Top‐left corner (i=0, j=0)
    lap[0,0] = (
        2 * field[0,0]
      + field[1,0]
      + field[0,1]
      - 4 * field[0,0]
    )
    # Top‐right corner (i=0, j=H-1)
    lap[0,-1] = (
        2 * field[0,-1]
      + field[1,-1]
      + field[0,-2]
      - 4 * field[0,-1]
    )
    # Bottom‐left corner (i=W-1, j=0)
    lap[-1,0] = (
        2 * field[-1,0]
      + field[-2,0]
      + field[-1,1]
      - 4 * field[-1,0]
    )
    # Bottom‐right corner (i=W-1, j=H-1)
    lap[-1,-1] = (
        2 * field[-1,-1]
      + field[-2,-1]
      + field[-1,-2]
      - 4 * field[-1,-1]
    )
    
    return lap

def update_diffusion(
    field: np.ndarray,
    secretion_map: np.ndarray,
    consumption_map: np.ndarray,
    D: float,
    decay: float,
    dt: float = P.delta_t
    ) -> np.ndarray:
    """
    Perform a single‐time‐step Euler update of the field array using:
        ∂C/∂t = D ∇2C − decay * C + secretion_map - consumption_map    (Eq. S1)
    - `field`         : current concentration array (shape: [W, H])
    - `secretion_map` : array of same shape giving Σ_a S_{i,a}(x,y) from all agents
    - `consumption_map` : array of field shape giving Σ_a U_{i,a}(x,y) from all agents
    - `D`             : diffusion coefficient (e.g., D_TGFb)
    - `decay`         : decay rate (e.g., λ_TGFb)
    - `dt`            : time increment (Δt)
    Returns the updated concentration array (clipped to ≥ 0).
    """
    
    # 1) Compute Laplacian (∇2C)
    lap = laplacian(field)
    
    # 2) Compute ∂C/∂t = D * lap − decay * field + secretion_map - consumption_map
    dCdt = D * lap - decay * field + secretion_map - consumption_map
    
    # 3) Euler step: C(t + dt) = C(t) + dt * dC/dt
    new_field = field + dt * dCdt
    
    # 4) Enforce non‐negativity
    new_field[new_field < 0] = 0.0
    
    return new_field

def update_diffusion_over_timestep(
    field: np.ndarray,
    secretion_map: np.ndarray,
    consumption_map: np.ndarray,
    D: float,
    decay: float,
    dt: float = P.delta_t,
    dT: float = P.timestep
    ) -> np.ndarray:
    """
    Advance `field` forward by macro-time-step `dT` seconds, using repeated calls
    to update_diffusion(...) with a micro‐time‐step of `dt`.
    """
    
    n_steps = int(dT / dt)
    
    # Perform that many Euler‐march diffusion updates
    for _ in range(n_steps):
        field = update_diffusion(field, secretion_map, consumption_map, D, decay, dt)
    return field