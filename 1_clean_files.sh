#!/bin/bash

# The code pre-processes the data:
#  - remove rows contain commas in a numerical field
#  - remove \N tag for None values

export LC_ALL=C
for f in data/raw/*.csv; do
    base=$(basename "$f" .csv)
    echo $f
    grep -v '"[^",][^"]*,[^"]*"' "$f" | sed 's/\\N//g' > "data/cleaned/${base}_cleaned.csv"
done