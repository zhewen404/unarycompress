DOTV_NAME=accumulator
RTL_NAME=accumulator
MODULE_NAME=accumulator
DIR_NAME=accumulator
ROOT_DIR=../..

# Default target
all: run

# Compile and run simulation
run: $(RTL_NAME)_sim
	./$(RTL_NAME)_sim

# Compile testbench
$(RTL_NAME)_sim: $(RTL_NAME).v $(RTL_NAME)_tb.v
	iverilog -o $(RTL_NAME)_sim $(RTL_NAME).v $(RTL_NAME)_tb.v

# Clean generated files
clean:
	rm -f $(RTL_NAME)_sim *.vcd

# preproc synthesis scripts
preproc:
	cp $(ROOT_DIR)/syn_scripts/run_synthesis.sh .
	cp $(ROOT_DIR)/syn_scripts/synthesis_area_opt.tcl .
	cp $(ROOT_DIR)/syn_scripts/.synopsys_dc.setup .
	cp $(ROOT_DIR)/syn_scripts/clean.sh .
	sed -i 's/\[dotv_name\]/$(DOTV_NAME)/g' run_synthesis.sh
	sed -i 's/\[dotv_name\]/$(DOTV_NAME)/g' synthesis_area_opt.tcl
	sed -i 's/\[rtl_name\]/$(RTL_NAME)/g' run_synthesis.sh
	sed -i 's/\[rtl_name\]/$(RTL_NAME)/g' synthesis_area_opt.tcl
	sed -i 's/\[module_name\]/$(MODULE_NAME)/g' run_synthesis.sh
	sed -i 's/\[module_name\]/$(MODULE_NAME)/g' synthesis_area_opt.tcl
	sed -i 's/\[dir_name\]/$(DIR_NAME)/g' run_synthesis.sh
	sed -i 's/\[dir_name\]/$(DIR_NAME)/g' synthesis_area_opt.tcl

# run synthesis after preprocessing
synth: preproc
	chmod +x run_synthesis.sh
	./run_synthesis.sh

.PHONY: all run clean synth preproc