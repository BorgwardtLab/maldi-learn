#!/usr/bin/env zsh
#
# Queries the data set for missing spectra. This operation is performed
# for all sites shipped with the data set.

for SITE in "DRIAMS-A" "DRIAMS-B" "DRIAMS-C" "DRIAMS-D"; do
  ROOT=/links/groups/borgwardt/Data/DRIAMS/$SITE/id
  echo $SITE
  for DIRECTORY in $ROOT/*; do
    YEAR=${DIRECTORY:t}
    ID_FILES=("${(@f)$(ls $DIRECTORY/${YEAR}*.csv)}")
    echo "  $YEAR"
    for ID_FILE in $ID_FILES; do
      echo "  $ID_FILE"
      for TYPE in "binned_6000" "preprocessed"; do
        echo "    $TYPE"
        python list_missing_spectra.py $ID_FILE /links/groups/borgwardt/Data/DRIAMS/$SITE/$TYPE/
      done
    done
  done
done
