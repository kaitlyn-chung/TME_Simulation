# test_model.py

#!/usr/bin/env python3
"""
Simple test script to verify the ABM model runs without errors.
"""

import numpy as np
import random
from ABM_Jul25.model import ABM_Model

def test_model_basic():
    """Test basic model initialization and running."""
    print("Testing ABM Model...")
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create a smaller model for testing
    model = ABM_Model(
        width=20,
        height=20,
        initial_tumor_cells=10,
        initial_CD8Tcells=5,
        initial_CD4Tcells=5,
        initial_macrophages=5,
        initial_MDSC=3
    )
    
    print(f"Model initialized with {len(model.schedule.agents)} agents")
    print(f"Grid size: {model.width}x{model.height}")
    
    # Import agent classes for counting
    from ABM_Jul25.agents import CancerCell, CD8TCell, CD4TCell, Macrophage, MDSC
    
    # Run for a few steps
    for step in range(5):
        print(f"Running step {step + 1}...")
        model.step()
        
        # Print agent counts
        cancer_count = model.count_agents(CancerCell)
        cd8_count = model.count_agents(CD8TCell)
        total_agents = len(model.schedule.agents)
        
        print(f"  Step {step + 1}: {total_agents} total agents ({cancer_count} cancer, {cd8_count} CD8)")
        
        if not model.running:
            print(f"  Model stopped running at step {step + 1}")
            break
    
    print("Model test completed successfully!")
    return model

def test_fields():
    """Test that diffusion fields are working."""
    print("\nTesting diffusion fields...")
    
    model = ABM_Model(width=10, height=10, initial_tumor_cells=5, 
                     initial_CD8Tcells=2, initial_CD4Tcells=2, 
                     initial_macrophages=2, initial_MDSC=1)
    
    # Check initial field states
    print(f"Initial TGFb field sum: {np.sum(model.TGFb_field)}")
    print(f"Initial IL2 field sum: {np.sum(model.IL2_field)}")
    
    # Run one step
    model.step()
    
    # Check field states after one step
    print(f"After 1 step TGFb field sum: {np.sum(model.TGFb_field)}")
    print(f"After 1 step IL2 field sum: {np.sum(model.IL2_field)}")
    
    print("Diffusion fields test completed!")

if __name__ == "__main__":
    try:
        model = test_model_basic()
        test_fields()
        print("\n✓ All tests passed! The model is working correctly.")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()