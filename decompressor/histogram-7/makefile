DOTV_NAME=histogram
MODULE_NAME=histogram_decompressor
DIR_NAME=histogram-7
ROOT_DIR=../..

# Default target
all: run

# Compile and run simulation
run: $(DOTV_NAME)_sim
	./$(DOTV_NAME)_sim

# Compile testbench
$(DOTV_NAME)_sim: $(DOTV_NAME).v $(DOTV_NAME)_tb.v
	iverilog -o $(DOTV_NAME)_sim $(DOTV_NAME).v $(DOTV_NAME)_tb.v

# Clean generated files
clean:
	rm -f $(DOTV_NAME)_sim *.vcd

# preproc synthesis scripts
preproc:
	cp $(ROOT_DIR)/syn_scripts/run_synthesis.sh .
	cp $(ROOT_DIR)/syn_scripts/synthesis_area_opt.tcl .
	cp $(ROOT_DIR)/syn_scripts/.synopsys_dc.setup .
	cp $(ROOT_DIR)/syn_scripts/clean.sh .
	sed -i 's/\[dotv_name\]/$(DOTV_NAME)/g' run_synthesis.sh
	sed -i 's/\[dotv_name\]/$(DOTV_NAME)/g' synthesis_area_opt.tcl
	sed -i 's/\[module_name\]/$(MODULE_NAME)/g' run_synthesis.sh
	sed -i 's/\[module_name\]/$(MODULE_NAME)/g' synthesis_area_opt.tcl
	sed -i 's/\[dir_name\]/$(DIR_NAME)/g' run_synthesis.sh
	sed -i 's/\[dir_name\]/$(DIR_NAME)/g' synthesis_area_opt.tcl

# run synthesis after preprocessing
synth: preproc
	chmod +x run_synthesis.sh
	./run_synthesis.sh
	@echo "Parsing area results..."
	@echo "Combinational area: $$(grep 'Combinational area:' reports/area_total.rpt | awk '{print $$3}')"
	@echo "Noncombinational area: $$(grep 'Noncombinational area:' reports/area_total.rpt | awk '{print $$3}')"
	@echo "Parsing power results..."
	@grep "^sequential" reports/power.rpt
	@grep "^combinational" reports/power.rpt
	@grep "^-*$$" reports/power.rpt | tail -1
	@grep "^Total.*mW$$" reports/power.rpt

.PHONY: all run clean synth preproc