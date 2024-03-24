python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "AQWan.csv" --strategy-args '{"pred_len":96}' --model-name "darts.RegressionModel" --gpus 0  --num-workers 1  --timeout 60000  --save-path "AQWan/darts_regressionmodel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "AQWan.csv" --strategy-args '{"pred_len":192}' --model-name "darts.RegressionModel" --gpus 0  --num-workers 1  --timeout 60000  --save-path "AQWan/regressionmodel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "AQWan.csv" --strategy-args '{"pred_len":336}' --model-name "darts.RegressionModel" --gpus 0  --num-workers 1  --timeout 60000  --save-path "AQWan/regressionmodel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "AQWan.csv" --strategy-args '{"pred_len":720}' --model-name "darts.RegressionModel" --gpus 0  --num-workers 1  --timeout 60000  --save-path "AQWan/regressionmodel"
