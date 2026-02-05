#!/bin/bash

#============================================================================
# Quick Synthesis Script for [module_name] Design
# Uses NanGate FreePDK45 library for area optimization
#============================================================================

echo "============================================================================"
echo "[module_name] Design - Area-Optimized Synthesis"
echo "Using NanGate FreePDK45 Library"
echo "============================================================================"

# Check if we're in the right directory
if [ ! -f "[dotv_name].v" ]; then
    echo "Error: [dotv_name].v not found!"
    echo "Please run this script from the [dir_name] directory"
    exit 1
fi

# Create necessary directories
mkdir -p reports outputs work

# Check for command line argument
if [ "$1" = "test" ]; then
    echo "Testing NanGate FreePDK45 library setup..."
    dc_shell-t -f test_library.tcl
    exit $?
fi

echo "Starting synthesis with Design Compiler..."
echo "Using NanGate FreePDK45 library"

# Run synthesis
dc_shell-t -f synthesis_area_opt.tcl

# Check if synthesis was successful
if [ $? -eq 0 ]; then
    echo "============================================================================"
    echo "SYNTHESIS COMPLETED SUCCESSFULLY"
    echo "============================================================================"
    echo ""
    echo "Generated Files:"
    echo "  Reports:  ./reports/"
    ls -la reports/ 2>/dev/null | head -10
    echo ""
    echo "  Outputs:  ./outputs/" 
    ls -la outputs/ 2>/dev/null
    echo ""
    echo "Key Results:"
    if [ -f "reports/area_total.rpt" ]; then
        echo "=== AREA SUMMARY ==="
        grep -A 10 "Total cell area" reports/area_total.rpt 2>/dev/null || echo "Area report not found"
    fi
    echo ""
    if [ -f "reports/timing_max.rpt" ]; then
        echo "=== CRITICAL PATH ==="
        grep -A 5 "slack" reports/timing_max.rpt 2>/dev/null | head -10 || echo "Timing report not found"
    fi
    echo "============================================================================"
else
    echo "============================================================================"
    echo "SYNTHESIS FAILED"
    echo "============================================================================"
    echo "Please check the error messages above and:"
    echo "1. Verify Design Compiler is in your PATH"
    echo "2. Check library setup in .synopsys_dc.setup"  
    echo "3. Review synthesis script for any issues"
    echo "============================================================================"
fi