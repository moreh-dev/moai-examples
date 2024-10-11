#! /bin/bash

cd /root/poc/inference_codes
python benchmark_client.py  --input-len 200 --output-len 200 --num-prompts 1  --num-trial 3 --save-result --result-dir benchmark_result --default-token 6176
python benchmark_client.py  --input-len 1024 --output-len 200 --num-prompts 1  --num-trial 3 --save-result --result-dir benchmark_result --default-token 6176
python benchmark_client.py  --input-len 8192 --output-len 200 --num-prompts 1  --num-trial 3 --save-result --result-dir benchmark_result --default-token 6176
python benchmark_client.py  --input-len 31768 --output-len 1000 --num-prompts 1  --num-trial 3 --save-result --result-dir benchmark_result --default-token 6176

