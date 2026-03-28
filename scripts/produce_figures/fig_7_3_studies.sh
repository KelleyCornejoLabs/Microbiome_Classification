#!/bin/bash

source setpaths.sh

TMP_DIR="tmp_fig_3_study"
mkdir -p $TMP_DIR

echo "Finding common bacteria species for all oral studies"
COMMON_COLS_ORAL_3_STUDY=$(python3 $OVERLAP_SCRIPT --files $MANGHI_FORMATTED $HYUHN_FORMATTED $EDLUND_FORMATTED | tail -n 1)

python3 -c "print('Found', len('$COMMON_COLS_ORAL_3_STUDY'.split(',')), 'common species')"
echo $COMMON_COLS_ORAL_3_STUDY
# actinomyces_naeslundii,actinomyces_oris,actinomyces_viscosus,aggregatibacter_actinomycetemcomitans,aggregatibacter_aphrophilus,aggregatibacter_segnis,atopobium_parvulum,bifidobacterium_dentium,bifidobacterium_longum,campylobacter_concisus,campylobacter_gracilis,campylobacter_rectus,campylobacter_showae,capnocytophaga_gingivalis,capnocytophaga_ochracea,cardiobacterium_hominis,corynebacterium_matruchotii,eikenella_corrodens,escherichia_coli,filifactor_alocis,fretibacterium_fastidiosum,fusobacterium_necrophorum,fusobacterium_nucleatum,fusobacterium_periodonticum,gemella_haemolysans,gemella_morbillorum,gemella_sanguinis,granulicatella_adiacens,haemophilus_haemolyticus,haemophilus_influenzae,haemophilus_parahaemolyticus,haemophilus_parainfluenzae,haemophilus_pittmaniae,kingella_kingae,lactobacillus_fermentum,lactobacillus_gasseri,lautropia_mirabilis,leptotrichia_buccalis,leptotrichia_goodfellowii,leptotrichia_hofstadii,leptotrichia_wadei,neisseria_cinerea,neisseria_elongata,neisseria_flavescens,neisseria_gonorrhoeae,neisseria_lactamica,neisseria_meningitidis,neisseria_polysaccharea,neisseria_sicca,neisseria_subflava,olsenella_uli,parascardovia_denticolens,parvimonas_micra,porphyromonas_gingivalis,prevotella_denticola,prevotella_intermedia,prevotella_melaninogenica,prevotella_oris,ralstonia_pickettii,rothia_aeria,rothia_dentocariosa,rothia_mucilaginosa,selenomonas_sputigena,simonsiella_muelleri,streptococcus_agalactiae,streptococcus_australis,streptococcus_cristatus,streptococcus_gordonii,streptococcus_infantis,streptococcus_mutans,streptococcus_parasanguinis,streptococcus_pseudopneumoniae,streptococcus_pyogenes,streptococcus_salivarius,streptococcus_sanguinis,streptococcus_sobrinus,streptococcus_thermophilus,streptococcus_vestibularis,tannerella_forsythia,treponema_denticola,veillonella_atypica,veillonella_dispar,veillonella_parvula

MANGHI_TRAIN="$MANGHI_DATA_DIR/clustered_80_train.csv"
MANGHI_TEST="$MANGHI_DATA_DIR/clustered_80_test.csv"
MANGHI_VALIDATE="$MANGHI_DATA_DIR/clustered_80_validation.csv"

THREE_STUDY_MODEL="$TMP_DIR/oral_classified_3_study"
THREE_STUDY_MODEL_SIMPLIFIED="$TMP_DIR/oral_classified_3_study_simplified"

HYUHN_CLASSIFIED_3_STUDY="$TMP_DIR/hyuhn_classified_3_study"
EDLUND_CLASSIFIED_3_STUDY="$TMP_DIR/edlund_classified_3_study"

echo "Training model..."
# python3 $STRATABIONN_SCRIPT -itr $MANGHI_TRAIN -ite $MANGHI_TEST -f $COMMON_COLS_ORAL_3_STUDY -p $THREE_STUDY_MODEL -ts

echo "Running classification..."
# python3 $STRATABIONN_SCRIPT -ite $HYUHN_FORMATTED -cl -p $THREE_STUDY_MODEL_SIMPLIFIED -out $HYUHN_CLASSIFIED_3_STUDY
# python3 $STRATABIONN_SCRIPT -ite $EDLUND_FORMATTED -cl -p $THREE_STUDY_MODEL_SIMPLIFIED -out $EDLUND_CLASSIFIED_3_STUDY

# python utilities.py 3_study_pacmap --base-class   $MANGHI_FORMATTED \
#                                    --test-1-class $HYUHN_CLASSIFIED_3_STUDY.csv \
#                                    --test-2-class $EDLUND_CLASSIFIED_3_STUDY.csv \
#                                    --common_cols  $COMMON_COLS_ORAL_3_STUDY

echo $THREE_STUDY_MODEL_SIMPLIFIED