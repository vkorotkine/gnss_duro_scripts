#!/bin/bash


DATA_DIR="/home/vassili/projects/gnss/data/fppp_1224/f9p_ppp_1224"
RNX2RTKP_APP="/home/vassili/projects/gnss/RTKLIB/bin/rnx2rtkp"
OUTPUT_DIR="/home/vassili/projects/gnss/gnss_duro_scripts/output/f9p_ppp_1224"
OUTPUT_FILE=${OUTPUT_DIR}/my_ppk.pos
PLOT_DIR=${OUTPUT_DIR}/plots

mkdir -p $OUTPUT_DIR
mkdir -p $PLOT_DIR

$RNX2RTKP_APP -x 2-k ${DATA_DIR}/ppk.conf -o $OUTPUT_FILE ${DATA_DIR}/rover.obs ${DATA_DIR}/tmg23590.20o ${DATA_DIR}/rover.nav

python3 inspect_ppk.py $OUTPUT_FILE --output $PLOT_DIR
