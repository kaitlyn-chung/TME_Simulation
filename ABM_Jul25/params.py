# params.py
# All ABM parameters are defined here

# Grid size
# assume x/y dimension of each grid is 50μm (~single cell dimension)
grid_width  = 100
grid_height = 100

# Time step (delta t) for PDE updates
delta_t = 1

# Time step for ABM update
timestep = 21600 # sec = 6h

# Diffusion coefficients (unit: grid²/sec)
D_Arg1 = 0.04    # = 100 μm²/sec
D_CCL2 = 0.02    # = 50 μm²/sec 
D_IFNg = 0.04    # = 100 μm²/sec 
D_IL2  = 0.04    # = 100 μm²/sec 
D_IL10 = 5.6e-4    # = 1.4 μm²/sec 
D_NO = 1.52    # = 3800 μm²/sec 
D_O2 = 1.12    # = 2800 μm²/sec 
D_TGFb  =  0.0104    # = 26 μm²/sec
D_VEGFA =  0.0116   # = 29 μm²/sec
D_IL12 = 0.026    # = 65 μm²/sec, estimation based on IL-2

# Decay rates (unit: 1/sec)
lambda_Arg1 = 2e-6  # 1/sec
lambda_CCL2 = 1.67e-5  # 1/sec
lambda_IFNg = 6.5e-5  # 1/sec
lambda_IL2 = 2.78e-5  # 1/sec
lambda_IL10 = 4.6e-5  # 1/sec
lambda_NO = 1.56e-3  # 1/sec
lambda_O2 = 1e-5  # 1/sec
lambda_TGFb  = 1.65e-4  # 1/sec
lambda_VEGFA = 1.65e-4  # 1/sec
lambda_IL12 = 3e-5  # 1/sec, estimation

# Molecular weights (unit: g/mol)
MW_Arg1 = 4.1e+4 # g/mol = Da
MW_CCL2 = 1.4e+4 # g/mol = Da
MW_IFNg = 3.49e+4 # g/mol = Da, homodimer
MW_IL2 = 1.5e+4 # g/mol = Da
MW_IL10 = 1.8e+4 # g/mol = Da
MW_NO = 30 # g/mol = Da
MW_O2 = 32 # g/mol = Da
MW_TGFb = 4.4e+4 # g/mol = Da
MW_VEGFA = 4.4e+4 # g/mol = Da
MW_IL12 = 7e+4 # g/mol = Da, heterodimer

# Release rates by mass (unit: g/sec)
Arg1_release_mass = 6.64e-9 # g/sec, by MDSC
CCL2_release_mass = 2.86e-20 # by cancer cells
IFNg_release_mass = 7.64e-19 # by cancer, M1, CD4T, CD8T
IL2_release_mass = 1.43e-19 # by CD8T
IL10_release_Treg_mass = 6.25e-23 # by Treg
IL10_release_M_mass = 6.25e-22 # by MDSC, M2 MF
NO_release_mass = 1.71e-16 # by MDSC
TGFb_release_Treg_mass = 6.1e-20 # by Treg
TGFb_release_M2_mass = 1.04e-20 # by M2
TGFb_release_cancer_mass = 1.06e-20 # by cancer
VEGFA_release_M2_mass = 1.97e-22 # by M2
VEGFA_release_cstem_mass = 1.27e-22 # by cancer stem cells
VEGFA_release_cancer_mass = 1.27e-21 # by other cancer cells
IL12_release_mass = 5e-19 # by M1

# Release rates by molecule (unit: molecules/sec)
Avogadro_No = 6.022e+23 # molecules per mole
Arg1_release = Arg1_release_mass * Avogadro_No / MW_Arg1
CCL2_release = CCL2_release_mass * Avogadro_No / MW_CCL2
IFNg_release = IFNg_release_mass * Avogadro_No / MW_IFNg
IL2_release  = IL2_release_mass * Avogadro_No / MW_IL2
IL10_release_Treg = IL10_release_Treg_mass * Avogadro_No / MW_IL10
IL10_release_M = IL10_release_M_mass * Avogadro_No / MW_IL10
NO_release = NO_release_mass * Avogadro_No / MW_NO
TGFb_release_Treg = TGFb_release_Treg_mass * Avogadro_No / MW_TGFb
TGFb_release_M2 = TGFb_release_M2_mass * Avogadro_No / MW_TGFb
TGFb_release_cancer = TGFb_release_cancer_mass * Avogadro_No / MW_TGFb
VEGFA_release_M2 = VEGFA_release_M2_mass * Avogadro_No / MW_VEGFA
VEGFA_release_cstem = VEGFA_release_cstem_mass * Avogadro_No / MW_VEGFA
VEGFA_release_cancer = VEGFA_release_cancer_mass * Avogadro_No / MW_VEGFA
IL12_release = IL12_release_mass * Avogadro_No / MW_IL12

# Consumption rates (unit: molecules/sec)
O2_uptake = 1e-5 # oxygen consumption by cells
VEGFA_uptake = 5e-5 # VEGFA uptake by endothelial cells
IFNg_uptake_cancer = 0.001 # IFNg consumption by cancer cells

# Cancer Cell parameters
CancerCell_stemGrowthRate = 2.9e-6 # /sec = 0.25/day, stem cell growth rate
CancerCell_asymDivProb = 0.95 # probability of cancer stem cell asymmetric division
CancerCell_progGrowthRate = 5.8e-6 # /sec = 0.5/day, progenitor growth rate
CancerCell_progDivMax = 9 # max divisions of progenitor
CancerCell_senescentDeathRate = 1.2e-5 # /sec = 1/day, senescent death rate
CancerCell_stemMoveProb = 0.8 # probability of stem migration per timestep
CancerCell_MoveProb = 1 # probability of non-stem migration per timestep

# CD8T Cell parameters
TCD8_lifespanMean = 111456 # sec = 1.29 days
TCD8_lifespanSD = 12960 # sec = 0.15 days
TCD8_moveProb = 1 # migration probability at each timestep
TCD8_prolif_IL2th_mass = 6.346e-2 # sec * g/L = 63460 sec * ng/mL
TCD8_prolif_IL2th = TCD8_prolif_IL2th_mass * 1.25e-10 * Avogadro_No / MW_IL2 # sec * molecules / grid
TCD8_div_Interval = 43200 # sec = 12h
TCD8_div_Limit = 4 # division limit is 4 times
k_TCD8_killing = 1.16e-5 # cell/sec = 1 cell/day

# Coefficients in Hill function of TCD8 inhibition
k_hill_MDSC_TCD8 = 2 # Half‐maximal inhibition = 2 MDSCs in neighborhood
n_hill_MDSC_TCD8 = 1 # no cooperative inhibition
IC50_Arg1_TCD8 = 5.3e+11 # molecules = 61.7 mU
IC50_NO_TCD8 = 7.5e-10 * 1.25e-10 * Avogadro_No # molecules = 7.5e-10 M

# CD4T Cell parameters
TCD4_lifespanMean = 111456 # sec = 1.29 days
TCD4_lifespanSD = 12960 # sec = 0.15 days
TCD4_Treg_frac = 0.2 # 20% of CD4 is Treg by default
TCD4_moveProb = 1 # migration probability at each timestep
TCD4_k_Th_diff_Treg = 2.55e-7 # /sec = 0.022 /day Th -> Treg differentiation rate
k_TCD4_div = 1.16e-5 # /sec = 1/day
TCD4_div_Interval = 43200 # sec = 12h
CD4_DIVISION_LIMIT = 4 # Add missing parameter
EC50_Arg1_Treg = 1.9e+11 # molecules = 22.1 mU Arg1
k_Arg1_Treg_div = 2.43e-05 # /sec = 2.1 times /day

# MDSC parameters
#MDSC_lifespanMean = 5.7888e+6 # sec = 67 days
MDSC_lifespanMean = 5.7888e+5 # sec = 67 days
MDSC_lifespanSD = 8.64e+4 # sec = 1 day
MDSC_moveProb = 0.1 # migration probability at each timestep

# Macrophage parameters
#Mac_lifespanMean = 4.32e+6 # sec = 50 days
Mac_lifespanMean = 4.32e+5 # sec = 50 days
Mac_lifespanSD = 8.64e+4 # sec = 1 day
Mac_moveProb = 0.5 # migration probability at each timestep
k_Mac_M1_pol = 3.215e-7 # /sec = 0.02/day, M2 to M1 polarization rate
k_Mac_M2_pol = 2.894e-6 # /sec = 0.25/day, M1 to M2 polarization rate
EC50_TGFb_M2 = 1.4e-10 * 1.25e-10 * Avogadro_No # molecules = 1.4e-10 M induction of M2 pol
EC50_IL10_M2 = 8e-12 * 1.25e-10 * Avogadro_No # molecules = 8e-12 M induction of M2 pol
EC50_IFNg_M1 = 2.9e-9 * 1.25e-10 * Avogadro_No # molecules = 2.9e-9 M induction of M1 pol
EC50_IL12_M1 = 1.4e-10 * 1.25e-10 * Avogadro_No # molecules = 1.4e-10 M induction of M1 pol
IC50_IL10_phago = 2.7e-10 * 1.25e-10 * Avogadro_No # molecules = 2.7e-10 M inhibition of M1 phagocytosis
n_hill_IL10_phago = 2 # hill-coefficient of inhibition
k_M1_phago = 5.787e-6 # /sec = 0.5/day

# Immune Cell Recruitment parameters
k_TCD8_rec = 2.5e-7 # molecules/sec, estimate
k_TCD4_rec = 1.25e-7 # molecules/sec, estimate
k_MDSC_rec_base = 2.43e-8 # molecules/sec = 0.0021/day
#k_MDSC_rec_max = 1.4e-5 # molecules/sec = 1.2/day max recruitment rate at saturated CCL2
k_MDSC_rec_max = 1.4e-7 # molecules/sec = 0.012/day max recruitment rate at saturated CCL2
EC50_CCL2_MDSC_rec = 5e-10 * 1.25e-10 * Avogadro_No # molecules = 5e-10 M
k_Mac_rec = 2.46e-7 # molecules/sec = 1.7e5 cell/mL/day