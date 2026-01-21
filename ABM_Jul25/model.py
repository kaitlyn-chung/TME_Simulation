# model.py

from mesa import Model
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import numpy as np
import math
import random

import agents as A
from grid_utils import update_diffusion_over_timestep
import params as P


class SimultaneousActivation:
    """
    A two‐phase activation scheduler for Mesa agents.
    In each tick:
      1) Call every agent's step() to compute "next state" (buffered).
      2) Call every agent's advance() to apply those buffered changes.
    """
    def __init__(self, model):
        self.model = model
        self.agents = []
        self.time = 0

    def add(self, agent):
        """Add an agent to the scheduler."""
        if agent not in self.agents:
            self.agents.append(agent)

    def remove(self, agent):
        """Remove an agent from the scheduler."""
        if agent in self.agents:
            self.agents.remove(agent)

    def step(self):
        """
        Execute one tick:
          1) Loop through agents in insertion order and call step().
          2) Then loop through agents in the same order and call advance().
          3) Clean up dead agents.
        """
        # Create a copy of the agents list to avoid modification during iteration
        agents_copy = self.agents.copy()
        
        # Phase 1: every agent "thinks" (buffers its actions)
        for agent in agents_copy:
            if hasattr(agent, 'step') and getattr(agent, 'alive', True):
                try:
                    agent.step()
                except Exception as e:
                    # If agent step fails, mark as dead
                    if hasattr(agent, 'alive'):
                        agent.alive = False
                    if hasattr(agent, 'pos'):
                        agent.pos = None
        
        # Phase 2: every agent "applies" its buffered actions
        for agent in agents_copy:
            if hasattr(agent, 'advance') and getattr(agent, 'alive', True):
                try:
                    agent.advance()
                except Exception as e:
                    # If agent advance fails, mark as dead
                    if hasattr(agent, 'alive'):
                        agent.alive = False
                    if hasattr(agent, 'pos'):
                        agent.pos = None

        # Phase 3: Clean up dead agents
        self.agents = [agent for agent in self.agents if getattr(agent, 'alive', True)]

        self.time += 1


class ABM_Model(Model):
    """
    ABM Model of TME evolution: ref. Zhang et al. (2024).
    - Maintains grids for agents and diffusible fields.
    - Executes agent.step() for all agents each tick.
    - Updates diffusible fields via reaction–diffusion
    """
    def __init__(
        self,
        width: int = P.grid_width,
        height: int = P.grid_height,
        initial_tumor_cells: int = 100,
        initial_CD8Tcells: int = 20,
        initial_CD4Tcells: int = 30,
        initial_macrophages: int = 30,
        initial_MDSC: int = 10
    ):
        super().__init__()
        self.width  = width
        self.height = height
        self.timestep = P.timestep
        
        # Initialize ID counter
        self._current_id = 0

        # 1) Create a SingleGrid: each grid contains only one cell
        self.grid = SingleGrid(width, height, torus=False)

        # 2) Scheduler: SimultaneousActivation ensures all agents read the same fields before updating them
        self.schedule = SimultaneousActivation(self)

        # 3) Initialize diffusible fields (2D NumPy arrays)
        self.Arg1_field     = np.zeros((width, height))
        self.CCL2_field     = np.zeros((width, height))
        self.IFNg_field     = np.zeros((width, height))
        self.IL2_field     = np.zeros((width, height))
        self.IL10_field     = np.zeros((width, height))
        self.IL12_field     = np.zeros((width, height))
        self.NO_field     = np.zeros((width, height))
        #self.O2_field     = np.zeros((width, height))
        self.TGFb_field     = np.zeros((width, height))
        self.VEGFA_field     = np.zeros((width, height))

        # 4) Initialize secretion maps (reset to zero each step)
        self.Arg1_secretion_map     = np.zeros((width, height))
        self.CCL2_secretion_map     = np.zeros((width, height))
        self.IFNg_secretion_map     = np.zeros((width, height))
        self.IL2_secretion_map     = np.zeros((width, height))
        self.IL10_secretion_map     = np.zeros((width, height))
        self.IL12_secretion_map     = np.zeros((width, height))
        self.NO_secretion_map     = np.zeros((width, height))
        #self.O2_secretion_map     = np.zeros((width, height))
        self.TGFb_secretion_map     = np.zeros((width, height))
        self.VEGFA_secretion_map     = np.zeros((width, height))
     
        # 5) Initialize consumption maps (reset to zero each step)
        self.Arg1_consumption_map     = np.zeros((width, height))
        self.CCL2_consumption_map     = np.zeros((width, height))
        self.IFNg_consumption_map     = np.zeros((width, height))
        self.IL2_consumption_map     = np.zeros((width, height))
        self.IL10_consumption_map     = np.zeros((width, height))
        self.IL12_consumption_map     = np.zeros((width, height))
        self.NO_consumption_map     = np.zeros((width, height))
        #self.O2_consumption_map     = np.zeros((width, height))
        self.TGFb_consumption_map     = np.zeros((width, height))
        self.VEGFA_consumption_map     = np.zeros((width, height))

        # 6) Place initial agents at random positions
        self.initialize_cells(
            initial_tumor_cells,
            initial_CD8Tcells,
            initial_CD4Tcells,
            initial_MDSC,
            initial_macrophages
        )

        # 7) DataCollector: track number of each agent type over time
        self.datacollector = DataCollector(
            model_reporters={
                "CancerCellCount": lambda m: m.count_agents(A.CancerCell),
                "CD8TCount":     lambda m: m.count_agents(A.CD8TCell),
                "CD4TCount":       lambda m: m.count_agents(A.CD4TCell),
                "MacCount":       lambda m: m.count_agents(A.Macrophage),
                "MDSCCount":       lambda m: m.count_agents(A.MDSC)
            }
        )

        self.running = True

    def count_agents(self, agent_type):
        """
        Count alive agents of the given type in the current schedule.
        """
        count = sum(
            1
            for agent in self.schedule.agents
            if isinstance(agent, agent_type) and getattr(agent, "alive", True)
        )
        return count

    def count_cell_type(self, agent_type, subtype=None):
        """
        Count agents of a specific type and optionally subtype.
        """
        count = 0
        for agent in self.schedule.agents:
            if isinstance(agent, agent_type) and getattr(agent, "alive", True):
                if subtype is None or getattr(agent, "subtype", None) == subtype:
                    count += 1
        return count
    
    def _next_id(self):
        """Return the next available ID for a new agent."""
        self._current_id += 1
        return self._current_id

    def safe_remove_agent(self, agent):
        """Safely remove an agent from grid and schedule."""
        try:
            if agent.pos is not None:
                self.grid.remove_agent(agent)
            agent.pos = None
            agent.alive = False
            if agent in self.schedule.agents:
                self.schedule.remove(agent)
        except Exception as e:
            # If removal fails, just mark as dead
            agent.alive = False
            agent.pos = None

    def step(self):
        """
        1) Zero out all secretion maps.
        2) Let each agent execute its step (secretion + movement + interactions).
        3) Update each diffusible field via reaction–diffusion (Eq. S1).
        4) Collect data, check stopping criteria.
        """
        # ---- 1) Clear secretion and consumption maps ----
        self.Arg1_secretion_map[:]     = 0.0
        self.CCL2_secretion_map[:]     = 0.0
        self.IFNg_secretion_map[:]     = 0.0
        self.IL2_secretion_map[:]      = 0.0
        self.IL10_secretion_map[:]     = 0.0
        self.IL12_secretion_map[:]     = 0.0
        self.NO_secretion_map[:]       = 0.0
        #self.O2_field[:]     = 0.0
        self.TGFb_secretion_map[:]     = 0.0
        self.VEGFA_secretion_map[:]    = 0.0
        
        self.Arg1_consumption_map[:]     = 0.0
        self.CCL2_consumption_map[:]     = 0.0
        self.IFNg_consumption_map[:]     = 0.0
        self.IL2_consumption_map[:]      = 0.0
        self.IL10_consumption_map[:]     = 0.0
        self.IL12_consumption_map[:]     = 0.0
        self.NO_consumption_map[:]       = 0.0
        #self.O2_consumption_map[:]     = 0.0
        self.TGFb_consumption_map[:]     = 0.0
        self.VEGFA_consumption_map[:]    = 0.0
        
        # ---- 2) Agent actions: secretion, intracellular ODEs, movement, interactions ----
        self.schedule.step()

        # ---- 3) Reaction–Diffusion updates for each field (Eq. S1) ----
        self._update_diffusion()

        # ---- 4) Recruit immune cells to empty grids
        self._recruitment()

        # ---- 5) Data collection and stopping criterion ----
        self.datacollector.collect(self)
        if self.count_agents(A.CancerCell) == 0:
            self.running = False

    def _recruitment(self):
        dt = self.timestep
        # Recruit CD8T, CD4T, MDSC, or macs to empty grids by individual rates
        for x in range(self.width):
            for y in range(self.height):
                if not self.grid.is_cell_empty((x, y)):
                    continue
        
                # 1) Compute each subtype's "per‐tick" rate at this (x,y)
                local_CCL2 = self.CCL2_field[x, y]
                p_CD8T_rec = 1.0 - math.exp(-P.k_TCD8_rec * dt)
                p_CD4T_rec = 1.0 - math.exp(-P.k_TCD4_rec * dt)
                p_Mac_rec = 1.0 - math.exp(-P.k_Mac_rec * dt)
                alpha_MDSC = P.k_MDSC_rec_base + P.k_MDSC_rec_max * local_CCL2 / (local_CCL2 + P.EC50_CCL2_MDSC_rec)
                p_MDSC_rec = 1.0 - math.exp(-alpha_MDSC * dt)
                
                # 2) Total "attempt" rate
                p_total = p_CD8T_rec + p_CD4T_rec + p_Mac_rec + p_MDSC_rec

                if p_total <= 0:
                    continue  # no recruitment possible
        
                # 3) Draw once: either "no recruit" or choose which type
                u = random.random()

                if u >= p_total:
                    continue  # no recruitment this tick at (x,y)
        
                # 4) Otherwise, pick which subtype by partitioning [0, p_total]
                threshold_cd8 = p_CD8T_rec
                threshold_cd4 = (p_CD8T_rec + p_CD4T_rec)
                threshold_mac = (p_CD8T_rec + p_CD4T_rec + p_Mac_rec)
                # (implicitly, if u < 1.0 it falls into MDSC interval next)
        
                if u < threshold_cd8:
                    print("threshold_cd8=", threshold_cd8)
                    print("u=",u)
                    # recruit CD8
                    new_cd8 = A.CD8TCell(self._next_id(), self, (x, y))
                    self.grid.place_agent(new_cd8, (x, y))
                    self.schedule.add(new_cd8)
        
                elif u < threshold_cd4:
                    print("threshold_cd4=", threshold_cd4)
                    print("u=",u)
                    # recruit CD4 (20% chance Treg, 80% Thelper)
                    if random.random() < P.TCD4_Treg_frac:
                        new_cd4 = A.CD4TCell(self._next_id(), self, (x, y), subtype=A.CD4TSubtype.CD4TREG)
                    else:
                        new_cd4 = A.CD4TCell(self._next_id(), self, (x, y), subtype=A.CD4TSubtype.CD4THELPER)
                    self.grid.place_agent(new_cd4, (x, y))
                    self.schedule.add(new_cd4)
        
                elif u < threshold_mac:
                    print("threshold_mac=", threshold_mac)
                    print("u=",u)
                    # recruit Macrophage (M1 by default)
                    new_mac = A.Macrophage(self._next_id(), self, (x, y), subtype=A.MacSubtype.M1)
                    self.grid.place_agent(new_mac, (x, y))
                    self.schedule.add(new_mac)
        
                else:
                    print("p_total=", p_total)
                    print("u=",u)
                    # recruit MDSC
                    new_mdsc = A.MDSC(self._next_id(), self, (x, y))
                    self.grid.place_agent(new_mdsc, (x, y))
                    self.schedule.add(new_mdsc)

    def _update_diffusion(self):
        # Arg1
        self.Arg1_field = update_diffusion_over_timestep(
            self.Arg1_field,
            self.Arg1_secretion_map,
            self.Arg1_consumption_map,
            P.D_Arg1,
            P.lambda_Arg1,
            P.delta_t,
            self.timestep
        )
        # CCL2
        self.CCL2_field = update_diffusion_over_timestep(
            self.CCL2_field,
            self.CCL2_secretion_map,
            self.CCL2_consumption_map,
            P.D_CCL2,
            P.lambda_CCL2,
            P.delta_t,
            self.timestep
        )
        # IFNg
        self.IFNg_field = update_diffusion_over_timestep(
            self.IFNg_field,
            self.IFNg_secretion_map,
            self.IFNg_consumption_map,
            P.D_IFNg,
            P.lambda_IFNg,
            P.delta_t,
            self.timestep
        )
        # IL2
        self.IL2_field = update_diffusion_over_timestep(
            self.IL2_field,
            self.IL2_secretion_map,
            self.IL2_consumption_map,
            P.D_IL2,
            P.lambda_IL2,
            P.delta_t,
            self.timestep
        )
        # IL10
        self.IL10_field = update_diffusion_over_timestep(
            self.IL10_field,
            self.IL10_secretion_map,
            self.IL10_consumption_map,
            P.D_IL10,
            P.lambda_IL10,
            P.delta_t,
            self.timestep
        )
        # IL12
        self.IL12_field = update_diffusion_over_timestep(
            self.IL12_field,
            self.IL12_secretion_map,
            self.IL12_consumption_map,
            P.D_IL12,
            P.lambda_IL12,
            P.delta_t,
            self.timestep
        )
        # NO
        self.NO_field = update_diffusion_over_timestep(
            self.NO_field,
            self.NO_secretion_map,
            self.NO_consumption_map,
            P.D_NO,
            P.lambda_NO,
            P.delta_t,
            self.timestep
        )
        # TGFb
        self.TGFb_field = update_diffusion_over_timestep(
            self.TGFb_field,
            self.TGFb_secretion_map,
            self.TGFb_consumption_map,
            P.D_TGFb,
            P.lambda_TGFb,
            P.delta_t,
            self.timestep
        )
        # VEGFA
        self.VEGFA_field = update_diffusion_over_timestep(
            self.VEGFA_field,
            self.VEGFA_secretion_map,
            self.VEGFA_consumption_map,
            P.D_VEGFA,
            P.lambda_VEGFA,
            P.delta_t,
            self.timestep
        )
        # O2
        # self.O2_field = update_diffusion_over_timestep(
        #     self.O2_field,
        #     self.O2_secretion_map,
        #     self.O2_consumption_map,
        #     P.D_O2,
        #     P.lambda_O2,
        #     P.delta_t,
        #     self.timestep
        # )
        

    def initialize_cells(
        self,
        initial_tumor_cells,
        initial_CD8Tcells,
        initial_CD4Tcells,
        initial_MDSC,
        initial_macrophages
    ):
        """
        Place initial agents in the grid:
          - Cluster tumor cells around a random center.
          - Other cell types are placed uniformly at random without overlap.
        """
        # Build and shuffle the list of all positions
        all_positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        random.shuffle(all_positions)
    
        # --- a) Tumor cells: cluster around a random center ---
        # 1) Choose a random center for the tumor cluster
        center = all_positions.pop()  # last element after shuffle
        cx, cy = center
    
        # 2) Sort all positions by distance² from the center
        #    Include the center itself first
        all_positions.append(center)
        sorted_positions = sorted(
            all_positions,
            key=lambda pos: (pos[0] - cx)**2 + (pos[1] - cy)**2
        )
    
        # 3) Place tumor cells in the nearest available positions
        placed = 0
        for pos in sorted_positions:
            if placed >= initial_tumor_cells:
                break
            if not self.grid.is_cell_empty(pos):
                continue
            cancer = A.CancerCell(self._next_id(), self, pos)
            self.grid.place_agent(cancer, pos)
            self.schedule.add(cancer)
            placed += 1
            # Remove used position from all_positions to avoid reuse
            if pos in all_positions:
                all_positions.remove(pos)
        
        # Re‐shuffle leftover positions before placing CD8, CD4, MDSC, Macs
        random.shuffle(all_positions)
    
        # --- b) CD8+ T cells ---
        for _ in range(min(initial_CD8Tcells, len(all_positions))):
            pos = all_positions.pop()
            tcell = A.CD8TCell(self._next_id(), self, pos)
            self.grid.place_agent(tcell, pos)
            self.schedule.add(tcell)
    
        # --- c) CD4+ T cells (20% Treg / 80% Thelper) ---
        for _ in range(min(initial_CD4Tcells, len(all_positions))):
            pos = all_positions.pop()
            subtype = A.CD4TSubtype.CD4TREG if random.random() < 0.20 \
                      else A.CD4TSubtype.CD4THELPER
            tcell = A.CD4TCell(self._next_id(), self, pos, subtype=subtype)
            self.grid.place_agent(tcell, pos)
            self.schedule.add(tcell)
    
        # --- d) MDSCs ---
        for _ in range(min(initial_MDSC, len(all_positions))):
            pos = all_positions.pop()
            mdsc = A.MDSC(self._next_id(), self, pos)
            self.grid.place_agent(mdsc, pos)
            self.schedule.add(mdsc)
    
        # --- e) Macrophages (M1 by default) ---
        for _ in range(min(initial_macrophages, len(all_positions))):
            pos = all_positions.pop()
            mac = A.Macrophage(self._next_id(), self, pos, subtype=A.MacSubtype.M1)
            self.grid.place_agent(mac, pos)
            self.schedule.add(mac)