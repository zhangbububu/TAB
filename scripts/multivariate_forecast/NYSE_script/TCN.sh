python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"pred_len":24}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 104, "n_epochs": 100, "output_chunk_length": 24}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "NYSE/darts_tcnmodel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"pred_len":36}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 104, "n_epochs": 100, "output_chunk_length": 36}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "NYSE/TCN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"pred_len":48}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 104, "n_epochs": 100, "output_chunk_length": 48}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "NYSE/TCN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"pred_len":60}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 104, "n_epochs": 100, "output_chunk_length": 60}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "NYSE/TCN"
