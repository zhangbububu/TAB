python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"pred_len":24}' --model-name "darts.RNNModel" --gpus 0  --num-workers 1  --timeout 60000  --save-path "Wike2000/darts_rnnmodel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"pred_len":36}' --model-name "darts.RNNModel" --gpus 0  --num-workers 1  --timeout 60000  --save-path "Wike2000/RNN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"pred_len":48}' --model-name "darts.RNNModel" --gpus 0  --num-workers 1  --timeout 60000  --save-path "Wike2000/RNN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"pred_len":60}' --model-name "darts.RNNModel" --gpus 0  --num-workers 1  --timeout 60000  --save-path "Wike2000/RNN"
