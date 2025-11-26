#!/bin/bash

mkdir -p compile_flags

make compile >> compile_flags/BM_Seq_Strassen.txt

make execute EXECUTE=build/BM_Seq_Strassen.exe 