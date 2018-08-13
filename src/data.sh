#!/bin/sh

python3 exe_download.py
wait

python3 exe_annotate_grasp.py
wait

python3 exe_positive.py
wait

python3 exe_snap.py positive_sub.txt
wait

python3 exe_pn.py
wait

python3 exe_seq_split.py
wait

chromosomes='chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY'
lengths='200 500 1000 2000 4000'

for chrom in $chromosomes
do
  for length in $lengths
  do
    python3 exe_seq_fetch.py $chrom $length
    wait
  done
done


for length in $lengths
do
  python3 exe_seq_merge.py $length
  wait
done


for length in $lengths
do
  python3 exe_seq_clean.py $length
  wait
done


for length in $lengths
do
  python3 exe_add_phenotype.py $length
  wait
done

python3 exe_regulomedb_local_all.py
wait

python3 exe_cross_validate_dataset.py
wait

# python3 exe_clean.py
