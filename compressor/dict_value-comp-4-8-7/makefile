DOTV_NAME=dict_value
MODULE_NAME=dict_value_compressor
DIR_NAME=dict_value-comp-4-8-7
ROOT_DIR=../..


# preproc synthesis scripts
preproc:
	cp $(ROOT_DIR)/syn_scripts/run_synthesis.sh .
	cp $(ROOT_DIR)/syn_scripts/synthesis_area_opt.tcl .
	cp $(ROOT_DIR)/syn_scripts/.synopsys_dc.setup .
	cp $(ROOT_DIR)/syn_scripts/clean.sh .
	chmod +x clean.sh
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

.PHONY: synth preproc