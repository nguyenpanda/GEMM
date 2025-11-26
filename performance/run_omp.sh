#!/bin/bash

for f in ./BM_ufunc/log/*.json; do
    python BM_Baseline.py "$f" --legend
done
