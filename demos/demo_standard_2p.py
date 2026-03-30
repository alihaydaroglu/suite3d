"""
Suite3D demo script for standard multi-plane 2-photon imaging data.

Two demo configurations are available:
- HC_demo: Generic/conservative parameters (original defaults)
- v1_demo: Experimental parameters validated on TC030 (March 2025)
            with quality metrics, parameter sweep, and duplication analysis

Usage:
    # Run with conservative defaults:
    python demos/HC_demo.py --data_dir /path/to/tifs --output_dir /path/to/output

    # Run with experimental TC030 params + sweep:
    python demos/v1_demo.py --data_dir /path/to/tifs --output_dir /path/to/output --sweep
"""

print(__doc__)
print("Please run either HC_demo.py or v1_demo.py directly.")
