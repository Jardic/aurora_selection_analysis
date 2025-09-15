#!/bin/bash

# Raw files
echo -en raw"\t" >> report.txt && zcat $1 | wc -l | sed 's/$/\/4/' | bc >> report.txt

# Adaptor trimming
cutadapt -a AGATCGGAAGAG -A AGATCGGAAGAG -j 6 -o file_1_trimmed.fastq -p file_2_trimmed.fastq  $1 $2
echo -en trimmed"\t" >> report.txt && cat file_1_trimmed.fastq | wc -l | sed 's/$/\/4/' | bc >> report.txt

# Read merging
fastq-join file_1_trimmed.fastq file_2_trimmed.fastq -o file_joined.fastq
echo -en joined"\t" >> report.txt && cat file_joined.fastqjoin | wc -l | sed 's/$/\/4/' | bc >> report.txt

rm file_1_trimmed.fastq
rm file_2_trimmed.fastq

# Unifying read orientation
cat file_joined.fastqjoin | fastx_barcode_splitter.pl --bcfile primers.txt --bol --prefix file_join_split- --suffix .fastq --mismatches 5 --partial 5
seqkit seq -r -p file_join_split-RV.fastq > file_join_split_RVrc.fastq
cat file_join_split-FW.fastq file_join_split_RVrc.fastq > sense.fastq
echo -en orientation_unified"\t" >> report.txt && cat sense.fastq | wc -l | sed 's/$/\/4/' | bc >> report.txt

rm file_joined.fastqjoin
rm file_joined.fastqun1
rm file_joined.fastqun2 

# Primer clipping
cutadapt -a AACACTAATGATCTGCCCGATG -j 6 -o sense_primer_clipped.fastq sense.fastq
echo -en primer_clipped"\t" >> report.txt && cat sense_primer_clipped.fastq | wc -l | sed 's/$/\/4/' | bc >> report.txt

 rm sense.fastq

# Get only seq lines from fastq
cat sense_primer_clipped.fastq | awk 'NR%4==2' > sense_primer_clipped.txt
rm sense_primer_clipped.fastq

# Pattern matching
egrep -o -f lib_design_regexp.txt sense_primer_clipped.txt > sense_matching.txt
echo -en pattern_matched"\t" >> report.txt && cat sense_matching.txt | wc -l >> report.txt

#rm sense_primer_clipped.txt

# Unique sequence read counting
cat sense_matching.txt | sort | uniq -c | awk {'print $1 "\t" $2'} | sort -nr -k 1 > preprocessed.tsv
echo -en unique_seqs_counted"\t" >> report.txt && cat preprocessed.tsv | wc -l >> report.txt

rm sense_matching.txt


rm file_join_split-FW.fastq
rm file_join_split-RV.fastq
rm file_join_split_RVrc.fastq
rm file_join_split-unmatched.fastq