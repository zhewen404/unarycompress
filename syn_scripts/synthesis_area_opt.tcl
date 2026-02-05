# Synthesis Script for [module_name] with Area Optimization
# Target: NanGate FreePDK45 Standard Cell Library
# Optimization: Area-focused

#============================================================================
# Setup and Library Configuration
#============================================================================
# Create output directories
file mkdir ./reports
file mkdir ./outputs

# NanGate FreePDK45 library configuration
set search_path ". /filespace/z/zpan52/repo/wisc25/FreePDK45/osu_soc/lib/files"
set target_library "gscl45nm.db"
set link_library "* gscl45nm.db"

#============================================================================
# Read Design Files
#============================================================================
# Read the synthesizable Verilog file
read_file -format verilog {[dotv_name].v}

# Set the top-level design - synthesize the compressor which is the main functionality
set current_design [module_name]
link

#============================================================================
# Design Constraints - Combinational Design (No Clock)
#============================================================================
# Since your design is purely combinational, we'll set timing constraints differently
# Create a relaxed virtual clock for area-focused synthesis
create_clock -name "vclk" -period 50.0

# Set relaxed input delays to prioritize area over timing
set_input_delay -clock vclk 1.0 [all_inputs]

# Set relaxed output delays to prioritize area over timing  
set_output_delay -clock vclk 1.0 [all_outputs]

# Set input drive strength (assume moderate drive)
set_driving_cell -lib_cell NAND2X1 [all_inputs]

# Set output loading (assume moderate load)
set_load 0.05 [all_outputs]

# Set maximum transition time for aggressive area optimization
set_max_transition 0.5 [current_design]

# Set maximum fanout for area optimization
set_max_fanout 32 [current_design]

# Set maximum capacitance (relaxed for area)
set_max_capacitance 0.5 [current_design]

#============================================================================
# Area Optimization Settings
#============================================================================
# Enable area optimization mode
set compile_ultra_ungroup_dw false
set compile_seqmap_propagate_constants true
set compile_delete_unloaded_sequential_cells false

# Enable aggressive area optimization
set_flatten true
set compile_enable_constant_propagation_with_no_boundary_opt true

# Area optimization settings - aggressive mode
set compile_ultra_ungroup_dw false
set compile_seqmap_propagate_constants true
set compile_delete_unloaded_sequential_cells false

# Enable aggressive area optimization
set_flatten true
set compile_enable_constant_propagation_with_no_boundary_opt true

# Additional area optimization settings
set compile_area_effort high
set compile_new_optimization true

# Advanced Area Optimization Settings (Ultra Enhanced)
#============================================================================
puts "Setting ultra-aggressive area optimization parameters..."

# Resource sharing and structure optimization
set hlo_share_effort high
set hlo_resource_implementation area
set hlo_resource_allocation constraint_driven

# Disable timing-driven optimizations for area focus
set compile_timing_high_effort_script false

# Enable sharing of operators and resources
set_register_merging [current_design] true
set_register_replication false

# Allow restructuring for area with ultra settings
set_structure true
set_balance false

# Ultra-aggressive optimization settings
set hlo_dw_prefer_ultra_area true
set compile_ultra_ungroup_small_hierarchies true
set compile_dont_touch_annotated_ports false
set compile_preserve_subdesign_interfaces false

# Boundary optimization for area
set compile_seqmap_enable_output_inversion true

#============================================================================
# First Compilation - Medium Effort
#============================================================================
puts "Starting first compilation with medium effort..."
compile -map_effort medium

# Check design after first compile
check_design
report_constraint -all_violators

#============================================================================
# Design Restructuring for Area Optimization
#============================================================================
# Uniquify and ungroup for better area optimization
uniquify -force
ungroup -all -flatten

#============================================================================
# Second Compilation - Ultra High Effort with Maximum Area Focus
#============================================================================
puts "Starting second compilation with ultra high effort and maximum area focus..."

# Set ultra aggressive area-focused compilation switches
set compile_area_effort high
set hlo_resource_implementation area
set hlo_transform_constant_multiplication true

# Set cost function to prioritize area over timing
set_cost_priority -delay

# Ultra compile with maximum area optimization
compile -map_effort high -area_effort high -boundary_optimization

# Third pass - incremental area optimization
puts "Running incremental area optimization..."
compile -incremental_mapping -map_effort high -area_effort high

# Check design after final compile
check_design
report_constraint -all_violators

#============================================================================
# Design Analysis and Reporting
#============================================================================
puts "Generating reports..."

# Area reports
report_area -hierarchy > ./reports/area_hierarchy.rpt
report_area -designware > ./reports/area_designware.rpt
report_area > ./reports/area_total.rpt

# Resource reports
report_resources -hierarchy > ./reports/resources.rpt

# Constraint reports
report_constraint -all_violators > ./reports/constraints_violated.rpt

# Timing reports (for combinational paths)
report_timing -delay_type max -max_paths 10 > ./reports/timing_max.rpt
report_timing -delay_type min -max_paths 10 > ./reports/timing_min.rpt

# Power estimation
report_power > ./reports/power.rpt

# Cell usage report
report_cell > ./reports/cell_usage.rpt

# Design statistics
report_design > ./reports/design_stats.rpt

#============================================================================
# Write Output Files
#============================================================================
puts "Writing output files..."

# Write synthesized netlist
write -format verilog -hierarchy -output ./outputs/[module_name]_synthesized.v

# Write gate-level Verilog with timing
write -format verilog -output ./outputs/[module_name].vg

# Write SDC constraints
write_sdc ./outputs/[module_name].sdc

# Write DDC format for further optimization
write -format ddc -hierarchy -output ./outputs/[module_name].ddc

#============================================================================
# Final Summary
#============================================================================
puts "============================================================================"
puts "SYNTHESIS COMPLETE"
puts "============================================================================"
puts "Design: [current_design]"
puts "Target Library: [get_attribute [get_libs] full_name]"
echo "Area Report:"
report_area
echo "Critical Path:"
report_timing -delay_type max -max_paths 1
puts "============================================================================"
puts "Output files written to ./outputs/"
puts "Reports written to ./reports/"
puts "============================================================================"

exit