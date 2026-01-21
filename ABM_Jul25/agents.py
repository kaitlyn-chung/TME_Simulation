# agents.py
# All cellular agents are defined here

from mesa import Agent
from enum import Enum, auto
import random
import numpy as np
import ABM_Jul25.params as P
import math


class CancerSubtype(Enum):
    STEM        = auto()
    PROGENITOR  = auto()
    SENESCENT   = auto()

class CancerCell(Agent):
    """
    - STEM cells divide at rate CancerCell_stemGrowthRate.  Each division is:
        • Asymmetric (prob = CancerCell_asymDivProb):  produces one STEM + one PROGENITOR.
        • Symmetric  (prob = 1 − CancerCell_asymDivProb): produces two STEM.
    - PROGENITOR cells divide at rate CancerCell_progGrowthRate, up to a maximum of Q divisions.
      After CancerCell_progDivMax divisions, a progenitor becomes SENESCENT.
    - SENESCENT cells do not divide; they die at rate CancerCell_senescentDeathRate.
    """
    def __init__(self, unique_id, model, pos, subtype=CancerSubtype.STEM):
        super().__init__(model)
        self.unique_id = unique_id
        self.pos = pos
        self.subtype = subtype
        self.alive = True

        # For PROGENITOR cells, track how many times they've divided so far
        # (once this hits P.Q, that agent switches to SENESCENT).
        self.divisions_done = 0

        # Optionally track "age" if needed; not strictly required here:
        self.age = 0

    def step(self):
        # Check if agent has a valid position
        if self.pos is None or not hasattr(self, 'pos'):
            return
            
        x, y = self.pos
        
        # Secrete TGFb, CCL2, IFNg
        self.model.TGFb_secretion_map[x, y] += P.TGFb_release_cancer * self.model.timestep
        self.model.CCL2_secretion_map[x, y] += P.CCL2_release * self.model.timestep
        self.model.IFNg_secretion_map[x, y] += P.IFNg_release * self.model.timestep
        
        # Consume IFNg
        self.model.IFNg_consumption_map[x, y] += P.IFNg_uptake_cancer * self.model.timestep

        if self.subtype == CancerSubtype.STEM:
            self._stem_behavior(x, y)
            
            # Secrete VEGFA at VEGFA_release_cstem rate (molecules/sec)
            self.model.VEGFA_secretion_map[x, y] += P.VEGFA_release_cstem * self.model.timestep

        elif self.subtype == CancerSubtype.PROGENITOR:
            self._progenitor_behavior(x, y)
            
            # Secrete VEGFA at VEGFA_release_cancer rate (molecules/sec)
            self.model.VEGFA_secretion_map[x, y] += P.VEGFA_release_cancer * self.model.timestep

        elif self.subtype == CancerSubtype.SENESCENT:
            self._senescent_behavior(x, y)
            
            # Secrete VEGFA at VEGFA_release_cancer rate (molecules/sec)
            self.model.VEGFA_secretion_map[x, y] += P.VEGFA_release_cancer * self.model.timestep

        # Migration (only if still alive and has valid position)
        if self.alive and self.pos is not None:
            self._migrate()

    def advance(self):
        """For SimultaneousActivation compatibility"""
        pass

    def _stem_behavior(self, x, y):
        """
        Stem cell divides at rate Rs = P.CancerCell_stemGrowthRate (units: 1/sec).
        In a timestep‐sized time step, probability of any division = P.CancerCell_stemGrowthRate * timestep.
        If division occurs:
          - With prob = CancerCell_asymDivProb: one daughter = STEM, one = PROGENITOR.
          - Else (prob = 1 - CancerCell_asymDivProb): both daughters = STEM.
        """

        # 1) Attempt a division event this tick
        if random.random() < P.CancerCell_stemGrowthRate * self.model.timestep:
            # 2) Choose symmetric vs. asymmetric
            if random.random() < P.CancerCell_asymDivProb:
                # ASYMMETRIC → produce 1 STEM, 1 PROGENITOR
                # (We reuse this agent as one daughter and spawn a new agent for the other)
                # Decide randomly which subtype this agent "stays as"
                if random.random() < 0.5:
                    # Keep this agent as STEM; spawn a PROGENITOR daughter
                    new_subtype = CancerSubtype.PROGENITOR
                    self.subtype = CancerSubtype.STEM
                else:
                    # Convert this agent into a PROGENITOR; spawn a STEM daughter
                    new_subtype = CancerSubtype.STEM
                    self.subtype = CancerSubtype.PROGENITOR
                # Create the "other" daughter at a random empty neighbor
                self._spawn_daughter(new_subtype, x, y)

            else:
                # SYMMETRIC STEM → both daughters are STEM
                # We keep this agent as STEM, and spawn one additional STEM
                self._spawn_daughter(CancerSubtype.STEM, x, y)
                # (the existing agent remains STEM)

    def _progenitor_behavior(self, x, y):
        """
        Progenitor cell divides at rate Rp = P.CancerCell_progGrowthRate (units: 1/sec), 
        up to P.CancerCell_progDivMax total divisions. 
        After it has divided P.CancerCell_progDivMax times,
        it switches to SENESCENT and stops dividing.
        """
        # 1) Check if it's already exhausted its allowed divisions
        if self.divisions_done >= P.CancerCell_progDivMax:
            # Convert to SENESCENT immediately
            self.subtype = CancerSubtype.SENESCENT
            return

        # 2) Attempt a division event this tick
        if random.random() < P.CancerCell_progGrowthRate * self.model.timestep:
            # Divide into two PROGENITOR daughters
            self.divisions_done += 1

            # Keep this agent as a renewed PROGENITOR (with updated count)
            # and spawn one more PROGENITOR
            self._spawn_daughter(CancerSubtype.PROGENITOR, x, y)

            # If this cell has now reached Q total divisions, turn it into SENESCENT
            if self.divisions_done >= P.CancerCell_progDivMax:
                self.subtype = CancerSubtype.SENESCENT

    def _senescent_behavior(self, x, y):
        """
        SENESCENT cells cannot proliferate. They die at rate P.CancerCell_senescentDeathRate (1/sec).
        """
        if random.random() < P.CancerCell_senescentDeathRate * self.model.timestep:
            # Mark as dead and remove from grid and schedule
            self.alive = False
            self.model.grid.remove_agent(self)
            self.pos = None  # Clear position
            try:
                self.model.schedule.remove(self)
            except ValueError:
                pass  # if already removed
            return

        # (Optional) Senescent cells could secrete TGFβ or other SASP factors:
        # self.model.tgfb_secretion_map[x, y] += P.SASP_rate

    def _spawn_daughter(self, subtype, x, y):
        """
        Helper: Place one daughter of the given subtype in a random empty neighbor.
        If no empty neighbor exists, no daughter is produced (contact inhibition).
        """
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        empty = [pos for pos in neighbors if self.model.grid.is_cell_empty(pos)]
        if not empty:
            return

        new_pos = random.choice(empty)
        daughter = CancerCell(self.model._next_id(), self.model, new_pos, subtype=subtype)
        # If subtype is PROGENITOR, initialize divisions_done = 0 by default.
        self.model.grid.place_agent(daughter, new_pos)
        self.model.schedule.add(daughter)

    def _migrate(self):
        """
        Move with probability p_move depending on subtype:
         - STEM:  p_move = 0.8
         - PROGENITOR or SENESCENT: p_move = 1.0
        If moving, pick one random Moore neighbor and step there if empty.
        """
        # Safety check for valid position
        if self.pos is None:
            return
            
        # Determine movement probability
        if self.subtype == CancerSubtype.STEM:
            p_move = P.CancerCell_stemMoveProb
        else:
            p_move = P.CancerCell_MoveProb

        # Decide if cell migrates in this tick
        if random.random() >= p_move:
            return  # no movement this timestep

        # Gather all Moore neighbors (8 directions) excluding current cell
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        # Filter to only the empty cells
        empty_neighbors = [pos for pos in neighbors if self.model.grid.is_cell_empty(pos)]
        if not empty_neighbors:
            return  # nowhere to move

        # Choose one empty neighbor at random and move there
        new_pos = random.choice(empty_neighbors)
        self.model.grid.move_agent(self, new_pos)
        self.pos = new_pos


class CD8TCell(Agent):
    """
    CD8+ TCell agent with:
      - Pre‐sampled lifespan (Normal draw).
      - IL-2–driven division after accumulating enough exposure.
      - Fixed‐interval division spacing (≥ 2 ticks = 12 h).
      - Maximum of 4 divisions.
      - Constant IL-2 secretion.
      - Fixed‐probability migration (no chemotaxis).
    """

    def __init__(self, unique_id, model, pos):
        super().__init__(model)
        self.unique_id = unique_id
        self.pos = pos

        # 1) Sample a "death‐time" (seconds) from N(mean, sd), clamp at ≥ 0.
        raw = random.gauss(P.TCD8_lifespanMean, P.TCD8_lifespanSD)
        self.lifespan = max(raw, 0.0)

        # 2) Initialize internal variables
        self.age = 0.0                        # in seconds
        self.IL2_accum = 0.0                  # accumulated IL-2 exposure (molecules·s)
        self.divisions_done = 0               # how many times this cell has divided
        self.last_div_tick = -9999            # scheduler.time of last division (initialize far in past)

        # 3) Track exhaustion (if you still want to implement killing/exhaustion)
        self.exhaustion = False

        # 4) Mark as alive
        self.alive = True

    def step(self):
        """
        Each ABM tick (duration = model.timestep seconds):
         1) Age + check for death by lifespan.
         2) Accumulate local IL-2; check for division conditions.
         3) If still alive, secrete IL-2.
         4) Move with probability TCELL_MIG_PROB.
         5) (Optional) Attempt to kill any co-located CancerCell via exhaustion‐based kill.
        """
        # Check if agent has a valid position
        if self.pos is None or not hasattr(self, 'pos'):
            return
            
        x, y = self.pos

        # ---- 1) Age and lifespan‐based death ----
        self.age += self.model.timestep
        if self.age >= self.lifespan:
            # Remove cell
            self.alive = False
            self.model.grid.remove_agent(self)
            self.pos = None
            try:
                self.model.schedule.remove(self)
            except ValueError:
                pass
            return

        # ---- 2) Accumulate IL-2 exposure and attempt division ----
        local_IL2 = self.model.IL2_field[x, y]  # (molecules/µm³)
        self.IL2_accum += local_IL2 * self.model.timestep

        # Check if we have enough IL-2 exposure AND enough time since last division
        # AND haven't exceeded division limit.
        current_tick = self.model.schedule.time
        ticks_since_last_div = current_tick - self.last_div_tick

        if (self.IL2_accum >= P.TCD8_prolif_IL2th
            and ticks_since_last_div >= P.TCD8_div_Interval
            and self.divisions_done < P.TCD8_div_Limit):
            # Attempt to divide
            self._divide_if_possible()

        # ---- 3) Secrete IL-2 at rate IL2_release for dt seconds ----
        self.model.IL2_secretion_map[x, y] += P.IL2_release * self.model.timestep

        # ---- 4) Migration (random walk) ----
        if random.random() < P.TCD8_moveProb and self.pos is not None:
            self._migrate()

        # ---- 5) Attempt to kill co-located cancer cells ----
        if self.pos is not None:
            self._attempt_kill()

    def advance(self):
        """For SimultaneousActivation compatibility"""
        pass

    def _divide_if_possible(self):
        """
        Spawn a daughter at a random empty Moore neighbor, reset IL-2 accumulation,
        update division count & last_div_tick.  If no empty neighbor, do nothing
        (contact inhibition)—the exposure remains accumulated (could divide next tick).
        """
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        empty = [pos for pos in neighbors if self.model.grid.is_cell_empty(pos)]
        if not empty:
            return  # no space to divide; IL2 accumulation stays until next tick

        new_pos = random.choice(empty)
        daughter = CD8TCell(self.model._next_id(), self.model, new_pos)
        # Inherit anything you want from parent?  Typically, a new cell starts fresh:
        #   - Sample its own lifespan (already done in __init__).
        #   - il2_accum = 0, divisions_done = 0, last_div_tick = -9999 by default.
        # So we do not copy exhaustion, age, etc.

        self.model.grid.place_agent(daughter, new_pos)
        self.model.schedule.add(daughter)

        # Reset parent's IL-2 accumulator and record division
        self.IL2_accum = 0.0
        self.divisions_done += 1
        self.last_div_tick = self.model.schedule.time

    def _migrate(self):
        """
        Choose a random Moore neighbor (9 neighbors) and move there if empty.
        """
        if self.pos is None:
            return
            
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        empty = [pos for pos in neighbors if self.model.grid.is_cell_empty(pos)]
        if not empty:
            return
        new_pos = random.choice(empty)
        self.model.grid.move_agent(self, new_pos)
        self.pos = new_pos

    def _attempt_kill(self):
        x, y = self.pos
        tcell_neigh_coords = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        
        for cx, cy in tcell_neigh_coords:
            # Check bounds
            if not (0 <= cx < self.model.width and 0 <= cy < self.model.height):
                continue
                
            target = self.model.grid[cx][cy]
            if isinstance(target, CancerCell) and target.alive:
                
                # Count CD8 and MDSCs in target cell neighborhood
                cancer_neigh = self.model.grid.get_neighborhood((cx, cy), moore=True, include_center=False)
                N_CD8 = 0
                N_MDSC = 0
                N_totNeighbor = len(cancer_neigh)

                for nx, ny in cancer_neigh:
                    if not (0 <= nx < self.model.width and 0 <= ny < self.model.height):
                        continue
                    occupant = self.model.grid[nx][ny]
                    if isinstance(occupant, CD8TCell) and occupant.alive:
                        N_CD8 += 1
                    elif isinstance(occupant, MDSC) and occupant.alive:
                        N_MDSC += 1

                ## Compute probability of killing
                local_Arg1 = self.model.Arg1_field[x, y]
                local_NO = self.model.NO_field[x, y]
                
                H_MDSC = 1.0 / (1.0 + (N_MDSC / P.k_hill_MDSC_TCD8)**P.n_hill_MDSC_TCD8) # Hill-inhibition of killing by MDSC
                H_Arg1 = local_Arg1 / (local_Arg1 + P.IC50_Arg1_TCD8) # Hill-inhibition by Arg1, n=1
                H_NO = local_NO / (local_NO + P.IC50_NO_TCD8) # Hill-inhibition by NO, n=1
                alpha = P.k_TCD8_killing * (N_CD8 / float(N_totNeighbor)) * (1.0 - H_MDSC) * (1.0 - H_Arg1) * (1.0 - H_NO)
                p_kill = 1.0 - math.exp(-self.model.timestep * alpha)
                
                if random.random() < p_kill:
                    target.alive = False
                    self.model.grid.remove_agent(target)
                    try:
                        self.model.schedule.remove(target)
                    except ValueError:
                        pass


class CD4TSubtype(Enum):
    CD4THELPER = auto()
    CD4TREG  = auto()
    
class CD4TCell(Agent):
    """
    CD4+ TCell agent with:
      - Thelper or Treg subtypes
      - Pre‐sampled lifespan (Normal draw).
      - Standard division time: once a day
      - Maximum of 4 divisions.
      - Arg1-induced Treg differentiation
      - Constant Arg1, TGFb secretion.
      - Fixed‐probability migration (no chemotaxis).
    """

    def __init__(self, unique_id, model, pos, subtype=None):
        super().__init__(model)
        self.unique_id = unique_id
        self.pos = pos

        # 1) Assign subtype: if not provided, 20% chance to be Treg
        if subtype is None:
            if random.random() < P.TCD4_Treg_frac:
                self.subtype = CD4TSubtype.CD4TREG
            else:
                self.subtype = CD4TSubtype.CD4THELPER
        else:
            self.subtype = subtype

        # 2) Sample a lifespan (seconds) from Normal(mean, sd), clamp at ≥ 0
        raw = random.gauss(P.TCD4_lifespanMean, P.TCD4_lifespanSD)
        self.lifespan = max(raw, 0.0)
        self.age = 0.0  # in seconds

        # 3) Division recording
        self.divisions_done = 0
        # Minimum number of ticks between divisions (24h / dt)
        #self.min_div_ticks = int((24 * 3600.0) / self.model.dt)
        self.last_div_tick = -9999  # far in the past so first division can happen after interval

        # 4) Alive flag
        self.alive = True

    def step(self):
        x, y = self.pos

        # ---- 1) Age-based death ----
        self.age += self.model.timestep
        if self.age >= self.lifespan:
            self.alive = False
            self.model.grid.remove_agent(self)
            try:
                self.model.schedule.remove(self)
            except ValueError:
                pass
            return

        # ---- 2) Treg differentiation ----
        if self.subtype == CD4TSubtype.CD4THELPER:
            p_Th_diff_Treg = 1 - math.exp(-P.TCD4_k_Th_diff_Treg * self.model.timestep)
            if random.random() <= p_Th_diff_Treg:
                self.subtype = CD4TSubtype.CD4TREG

        # ---- 3) Proliferation (induced by Arg1, at most once per 24h, max 4 divisions) ----
        local_Arg1 = self.model.Arg1_field[x, y]
        current_tick = self.model.schedule.time
        ticks_since_div = current_tick - self.last_div_tick
        
        if self.subtype == CD4TSubtype.CD4THELPER:
            p_div = 1 - math.exp(-P.k_TCD4_div * self.model.timestep)
            if (ticks_since_div >= P.TCD4_div_Interval
                and self.divisions_done < P.CD4_DIVISION_LIMIT
                and random.random() <= p_div
                ):
                # Attempt to divide
                self._divide_if_possible()
                
        elif self.subtype == CD4TSubtype.CD4TREG:
            # Arg1-induced Treg proliferation
            p_div = 1 - math.exp( -P.k_Arg1_Treg_div * self.model.timestep * local_Arg1 / (local_Arg1 + P.EC50_Arg1_Treg) )
            if (ticks_since_div >= P.TCD4_div_Interval
                and self.divisions_done < P.CD4_DIVISION_LIMIT
                and random.random() <= p_div
                ):
                # Attempt to divide
                self._divide_if_possible()

        # ---- 4) Secretion ----
        if self.subtype == CD4TSubtype.CD4THELPER:
            # Thelper secretes IL-2 and IFNg
            self.model.IL2_secretion_map[x, y] += P.IL2_release * self.model.timestep
            self.model.IFNg_secretion_map[x, y] += P.IFNg_release * self.model.timestep
        elif self.subtype == CD4TSubtype.CD4TREG:
            # Treg secretes IL10, Arg1 and TGFb
            self.model.IL10_secretion_map[x, y] += P.IL10_release_Treg * self.model.timestep
            self.model.Arg1_secretion_map[x, y] += P.Arg1_release * self.model.timestep
            self.model.TGFb_secretion_map[x, y] += P.TGFb_release_Treg * self.model.timestep

        # ---- 5) Migration (random walk) ----
        if random.random() < P.TCD4_moveProb:
            self._migrate()

    def advance(self):
        """For SimultaneousActivation compatibility"""
        pass

    def _divide_if_possible(self):
        """
        If there's at least one empty Moore neighbor, spawn a daughter with same subtype.
        Reset division clock and increment divisions_done. If no empty neighbor, skip.
        """
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        empty = [pos for pos in neighbors if self.model.grid.is_cell_empty(pos)]
        if not empty:
            return  # no space to divide

        new_pos = random.choice(empty)
        daughter = CD4TCell(self.model._next_id(), self.model, new_pos, subtype=self.subtype)
        self.model.grid.place_agent(daughter, new_pos)
        self.model.schedule.add(daughter)

        self.divisions_done += 1
        self.last_div_tick = self.model.schedule.time

    def _migrate(self):
        """
        Choose a random Moore neighbor and move there if empty.
        """
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        empty = [pos for pos in neighbors if self.model.grid.is_cell_empty(pos)]
        if not empty:
            return
        new_pos = random.choice(empty)
        self.model.grid.move_agent(self, new_pos)
        self.pos = new_pos


class MDSC(Agent):
    """
    MDSC agent with:
      - Pre‐sampled lifespan (Normal draw).
      - Constant Arg1, NO, IL10 secretion.
      - Fixed‐probability migration (no chemotaxis).
    """

    def __init__(self, unique_id, model, pos):
        super().__init__(model)
        self.unique_id = unique_id
        self.pos = pos

        # 1) Sample a lifespan (seconds) from Normal(mean, sd), clamp at ≥ 0
        raw = random.gauss(P.MDSC_lifespanMean, P.MDSC_lifespanSD)
        self.lifespan = max(raw, 0.0)
        self.age = 0.0  # in seconds

        # 2) Alive flag
        self.alive = True

    def step(self):
        x, y = self.pos

        # ---- 1) Age-based death ----
        self.age += self.model.timestep
        if self.age >= self.lifespan:
            self.alive = False
            self.model.grid.remove_agent(self)
            try:
                self.model.schedule.remove(self)
            except ValueError:
                pass
            return

        # ---- 2) Secretion ----
        self.model.Arg1_secretion_map[x, y] += P.Arg1_release * self.model.timestep
        self.model.IL10_secretion_map[x, y] += P.IL10_release_M * self.model.timestep
        self.model.NO_secretion_map[x, y] += P.NO_release * self.model.timestep

        # ---- 3) Migration (random walk) ----
        if random.random() < P.MDSC_moveProb:
            self._migrate()

    def advance(self):
        """For SimultaneousActivation compatibility"""
        pass
            
    def _migrate(self):
        """
        Choose a random Moore neighbor and move there if empty.
        """
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        empty = [pos for pos in neighbors if self.model.grid.is_cell_empty(pos)]
        if not empty:
            return
        new_pos = random.choice(empty)
        self.model.grid.move_agent(self, new_pos)
        self.pos = new_pos


class MacSubtype(Enum):
    M1 = auto()
    M2 = auto()

class Macrophage(Agent):
    """
    Macrophage agent with:
      1) Two subtypes: M1 and M2
      2) Pre-sampled lifespan from Normal(mean, sd)
      3) M1↔M2 polarization based on local cytokines
      4) M1 phagocytoses neighboring CancerCell agents
      5) M1 secretes IL-12 and IFNγ; M2 secretes TGFβ and IL-10
      6) Moves randomly with probability 0.5 per tick
    """

    def __init__(self, unique_id, model, pos, subtype=None):
        super().__init__(model)
        self.unique_id = unique_id
        self.pos = pos

        # 1) Assign subtype: default to M1 unless specified
        if subtype is None:
            self.subtype = MacSubtype.M1
        else:
            self.subtype = subtype

        # 2) Sample a lifespan (seconds) from Normal(mean, sd), clamp ≥ 0
        raw = random.gauss(P.Mac_lifespanMean, P.Mac_lifespanSD)
        self.lifespan = max(raw, 0.0)
        self.age = 0.0  # in seconds

        # 3) Alive flag
        self.alive = True

    def step(self):
        x, y = self.pos
        dt = self.model.timestep

        # --- 1) Age‐based death ---
        self.age += dt
        if self.age >= self.lifespan:
            self.alive = False
            self.model.grid.remove_agent(self)
            try:
                self.model.schedule.remove(self)
            except ValueError:
                pass
            return

        # --- 2) Polarization ---
        self._polarization()

        # --- 3) Phagocytosis (only M1) ---
        if self.subtype == MacSubtype.M1:
            self._attempt_kill()

        # --- 4) Secretion ---
        if self.subtype == MacSubtype.M1:
            # M1 secretes IL-12 and IFNγ
            self.model.IL12_secretion_map[x, y] += P.IL12_release * dt
            self.model.IFNg_secretion_map[x, y] += P.IFNg_release * dt
        else:
            # M2 secretes VEGFA, TGFβ and IL-10
            self.model.VEGFA_secretion_map[x, y] += P.VEGFA_release_M2 * dt
            self.model.TGFb_secretion_map[x, y] += P.TGFb_release_M2 * dt
            self.model.IL10_secretion_map[x, y] += P.IL10_release_M * dt

        # --- 5) Migration (random walk) ---
        if random.random() < P.Mac_moveProb:
            self._migrate()

    def advance(self):
        """For SimultaneousActivation compatibility"""
        pass
            
    def _polarization(self):
        x, y = self.pos
        dt = self.model.timestep
        
        # M1 → M2 rate increases with local TGFβ and IL-10
        # M2 → M1 rate increases with local IL-12 and IFNγ

        local_TGFb = self.model.TGFb_field[x, y]
        local_IL10 = self.model.IL10_field[x, y]
        local_IL12 = self.model.IL12_field[x, y]
        local_IFNg = self.model.IFNg_field[x, y]

        if self.subtype == MacSubtype.M1:
            # Compute instantaneous rate r_M1→M2:
            MM_TGFb = local_TGFb / (local_TGFb + P.EC50_TGFb_M2)
            MM_IL10 = local_IL10 / (local_IL10 + P.EC50_IL10_M2)
            alpha_M2 = P.k_Mac_M2_pol * MM_TGFb * MM_IL10

            # Convert to probability
            p_M1_to_M2 = 1.0 - math.exp(-alpha_M2 * dt)
            if random.random() < p_M1_to_M2:
                self.subtype = MacSubtype.M2
                # Skip M1 behaviors this tick if switching
                return

        elif self.subtype == MacSubtype.M2:  # currently M2
            # Compute instantaneous rate r_M2→M1:
            MM_IL12 = local_IL12 / (local_IL12 + P.EC50_IL12_M1)
            MM_IFNg = local_IFNg / (local_IFNg + P.EC50_IFNg_M1)
            alpha_M1 = P.k_Mac_M1_pol * MM_IL12 * MM_IFNg
                
            p_M2_to_M1 = 1.0 - math.exp(-alpha_M1 * dt)
            if random.random() < p_M2_to_M1:
                self.subtype = MacSubtype.M1
                # Skip M2 behaviors this tick if switching
                return

    def _attempt_kill(self):
        x, y = self.pos
        M1_neigh_coords = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        
        for cx, cy in M1_neigh_coords:
            # Check bounds
            if not (0 <= cx < self.model.width and 0 <= cy < self.model.height):
                continue
                
            target = self.model.grid[cx][cy]
            if isinstance(target, CancerCell) and target.alive: # found target
                
                # Count M1s in target cell neighborhood
                cancer_neigh = self.model.grid.get_neighborhood((cx, cy), moore=True, include_center=True)
                N_M1 = 0
                N_totNeighbor = len(cancer_neigh)

                for nx, ny in cancer_neigh:
                    if not (0 <= nx < self.model.width and 0 <= ny < self.model.height):
                        continue
                    occupant = self.model.grid[nx][ny]
                    if isinstance(occupant, Macrophage) and occupant.subtype == MacSubtype.M1 and occupant.alive:
                        N_M1 += 1

                ## Compute probability of killing
                local_IL10 = self.model.IL10_field[x, y]
                
                H_IL10 = 1.0 / (1.0 + (P.IC50_IL10_phago / local_IL10)**P.n_hill_IL10_phago) # Hill-inhibition of phagocytosis by IL10
                alpha = P.k_M1_phago * (N_M1 / float(N_totNeighbor)) * (1.0 - H_IL10)
                p_kill = 1.0 - math.exp(-self.model.timestep * alpha)
                
                if random.random() < p_kill:
                    target.alive = False
                    self.model.grid.remove_agent(target)
                    try:
                        self.model.schedule.remove(target)
                    except ValueError:
                        pass

    def _migrate(self):
        """
        Move to a random empty Moore neighbor if available.
        """
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        empty = [pos for pos in neighbors if self.model.grid.is_cell_empty(pos)]
        if not empty:
            return
        new_pos = random.choice(empty)
        self.model.grid.move_agent(self, new_pos)
        self.pos = new_pos