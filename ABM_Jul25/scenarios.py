#!/usr/bin/env python3
"""
Possible scenarios for initializing ABM model.
"""
steps = width = height = int_tumor_cells = int_cd8 = int_cd4 = int_macrophage = int_mdsc = 'TBD'

# Default simulation scenarios
DEFAULT_SCENARIOS = {
    'small_test': {
        'steps': 20,
        'width': 20,
        'height': 20,
        'initial_tumor_cells': 20,
        'initial_CD8Tcells': 5,
        'initial_CD4Tcells': 8,
        'initial_macrophages': 8,
        'initial_MDSC': 3
    },
    'standard': {
        'steps': 100,
        'width': 30,
        'height': 30,
        'initial_tumor_cells': 80,
        'initial_CD8Tcells': 20,
        'initial_CD4Tcells': 30,
        'initial_macrophages': 30,
        'initial_MDSC': 10
    },
    'large_tumor': {
        'steps': 150,
        'width': 50,
        'height': 50,
        'initial_tumor_cells': 200,
        'initial_CD8Tcells': 40,
        'initial_CD4Tcells': 60,
        'initial_macrophages': 60,
        'initial_MDSC': 20
    },
    'immune_rich': {
        'steps': 100,
        'width': 30,
        'height': 30,
        'initial_tumor_cells': 30,
        'initial_CD8Tcells': 40,
        'initial_CD4Tcells': 50,
        'initial_macrophages': 40,
        'initial_MDSC': 5
    },
    'custom': {
        'steps': steps,
        'width': width,
        'height': height,
        'initial_tumor_cells': int_tumor_cells,
        'initial_CD8Tcells': int_cd8,
        'initial_CD4Tcells': int_cd4,
        'initial_macrophages': int_macrophage,
        'initial_MDSC': int_mdsc   
    }
}

def confirm_value(name: str, value) -> bool:
    name.lower()
    ans = input(f"You entered {value} for {name}. Enter 'Y' to confirm, anything else to retry: ").strip().upper()
    return ans == "Y"

def get_value(name: str, prompt: str, min_value: int = 1, confirm: bool = True) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print(f"{name} must be a whole number (e.g., 2, 10).")
            continue

        if value < min_value:
            print(f"{name} must be >= {min_value}.")
            continue

        if confirm and not confirm_value(name, value):
            print("Okay — let's try again.")
            continue

        return value