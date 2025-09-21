#!/usr/bin/env python3
"""
Validation scenario runner script.

Usage:
    python scripts/validate_scenario.py --scenario V1 --config config/edgeworth.yaml
    python scripts/validate_scenario.py --all --output results/validation/
"""

import argparse
import sys
import subprocess
from pathlib import Path

def run_pytest_scenario(scenario: str = None) -> bool:
    """Run validation scenarios using pytest."""
    cmd = ["python", "-m", "pytest", "tests/validation/", "-v"]
    
    if scenario:
        # Map scenario names to pytest test names
        scenario_map = {
            "V1": "test_v1_edgeworth_2x2",
            "V2": "test_v2_spatial_null", 
            "V3": "test_v3_market_access",
            "V4": "test_v4_throughput_cap",
            "V5": "test_v5_spatial_dominance",
            "V6": "test_v6_price_normalization",
            "V7": "test_v7_empty_marketplace",
            "V8": "test_v8_stop_conditions",
            "V9": "test_v9_scale_invariance",
            "V10": "test_v10_spatial_null_unit"
        }
        
        if scenario in scenario_map:
            cmd.extend(["-k", scenario_map[scenario]])
        else:
            print(f"Error: Unknown scenario {scenario}. Valid scenarios: {list(scenario_map.keys())}")
            return False
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Validation failed: {e}")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run validation scenarios")
    parser.add_argument("--scenario", help="Specific scenario to run (V1-V10)")
    parser.add_argument("--config", help="Configuration file (currently ignored - uses built-in test configs)")
    parser.add_argument("--all", action="store_true", help="Run all validation scenarios")
    parser.add_argument("--output", help="Output directory (currently ignored - pytest handles output)")
    
    args = parser.parse_args()
    
    if not (args.scenario or args.all):
        print("Error: Must specify --scenario or --all")
        sys.exit(1)
    
    if args.config:
        print("Note: --config option not yet implemented. Using built-in test configurations.")
    
    if args.output:
        print("Note: --output option not yet implemented. Using pytest default output.")
    
    print("Running validation scenarios via pytest...")
    
    if args.all:
        print("Running all validation scenarios (V1-V10)...")
        success = run_pytest_scenario()
    else:
        print(f"Running validation scenario: {args.scenario}")
        success = run_pytest_scenario(args.scenario)
    
    if success:
        print("✅ Validation completed successfully!")
        sys.exit(0)
    else:
        print("❌ Validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()