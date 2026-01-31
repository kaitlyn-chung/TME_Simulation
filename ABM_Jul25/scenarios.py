#!/usr/bin/env python3
"""
Possible scenarios for initializing ABM model.
"""

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
    }
}