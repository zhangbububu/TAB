python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"pred_len":24}' --model-name "time_series_library.Linear" --model-hyper-params '{"batch_size": 16, "d_ff": 512, "d_model": 256, "lr": 0.01, "pred_len": 24, "seq_len": 104}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ILI/Linear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"pred_len":36}' --model-name "time_series_library.Linear" --model-hyper-params '{"batch_size": 8, "d_ff": 512, "d_model": 256, "lr": 0.01, "pred_len": 36, "seq_len": 104}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ILI/Linear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"pred_len":48}' --model-name "time_series_library.Linear" --model-hyper-params '{"batch_size": 8, "d_ff": 512, "d_model": 256, "lr": 0.01, "pred_len": 48, "seq_len": 104}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ILI/Linear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"pred_len":60}' --model-name "time_series_library.Linear" --model-hyper-params '{"batch_size": 8, "d_ff": 512, "d_model": 256, "lr": 0.01, "pred_len": 60, "seq_len": 104}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ILI/Linear"

