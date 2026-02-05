DOTV_NAME=dict_value
MODULE_NAME=dict_value_compressor_with_reg
DIR_NAME=dict_value-4-8-7
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

.PHONY: synth preproc