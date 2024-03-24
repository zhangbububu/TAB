python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"pred_len":96}' --model-name "self_implementation.VAR_model" --gpus 0  --num-workers 1  --timeout 60000  --save-path "Traffic/VAR_model"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"pred_len":192}' --model-name "self_implementation.VAR_model" --gpus 0  --num-workers 1  --timeout 60000  --save-path "Traffic/VAR"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"pred_len":336}' --model-name "self_implementation.VAR_model" --gpus 0  --num-workers 1  --timeout 60000  --save-path "Traffic/VAR"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"pred_len":720}' --model-name "self_implementation.VAR_model" --gpus 0  --num-workers 1  --timeout 60000  --save-path "Traffic/VAR"
