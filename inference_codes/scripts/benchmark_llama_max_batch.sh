#! /bin/bash

cd /root/poc/inference_codes
python benchmark_client.py  --input-len 200 --output-len 200 --num-prompts 256  --num-trial 3 --save-result --result-dir benchmark_result
python benchmark_client.py  --input-len 1024 --output-len 200 --num-prompts 256  --num-trial 3 --save-result --result-dir benchmark_result
python benchmark_client.py  --input-len 4096 --output-len 200 --num-prompts 85  --num-trial 3 --save-result --result-dir benchmark_result
python benchmark_client.py  --input-len 7992 --output-len 200 --num-prompts 43  --num-trial 3 --save-result --result-dir benchmark_result