#!/usr/bin/env zsh
#
# Performs binning for all DRIAMS sites. Spectra will never be
# overwritten by this script.

# We want `6000` to finish first for all sites before starting with
# another bin size.
for BINS in 6000 3600; do
  for SITE in "DRIAMS-A" "DRIAMS-B" "DRIAMS-C" "DRIAMS-D" "DRIAMS-E" "DRIAMS-F"; do
    ROOT=/links/groups/borgwardt/Data/DRIAMS/$SITE/id
    echo $SITE
    for DIRECTORY in $ROOT/*; do
      YEAR=${DIRECTORY:t}
      python bin_driams.py -s $SITE -y $YEAR -b $BINS
    done
  done
done
