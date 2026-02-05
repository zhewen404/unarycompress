#!/bin/bash

# Clean up script for synthesis directory
# Usage: ./clean.sh [all|temp|reports]

echo "============================================================================"
echo "Synthesis - Cleanup Script"
echo "============================================================================"

case "$1" in
    "all")
        echo "Removing ALL generated files (keeping only source RTL and scripts)..."
        rm -rf outputs reports cksum_dir work
        rm -f *.mr *.pvl *.syn *.svf *.log command.log
        echo "Cleaned: All synthesis outputs removed"
        ;;
    "temp")
        echo "Removing temporary synthesis files..."
        rm -f *.mr *.pvl *.syn *.svf *.log command.log
        rm -rf cksum_dir work
        echo "Cleaned: Temporary files removed"
        ;;
    "reports")
        echo "Removing only reports directory..."
        rm -rf reports
        echo "Cleaned: Reports directory removed"
        ;;
    *)
        echo "Usage: $0 [all|temp|reports]"
        echo ""
        echo "Options:"
        echo "  all      - Remove all generated files (outputs, reports, temp files)"
        echo "  temp     - Remove only temporary synthesis files (.mr, .pvl, .syn, etc.)"
        echo "  reports  - Remove only the reports directory"
        echo ""
        echo "Files that will be KEPT:"
        echo "  - *.v files (RTL source)"
        echo "  - *.tcl files (synthesis scripts)"  
        echo "  - run_synthesis.sh (synthesis runner)"
        echo "  - .synopsys_dc.setup (library setup)"
        echo "  - README.md (documentation)"
        echo ""
        echo "Current directory contents:"
        ls -la
        ;;
esac