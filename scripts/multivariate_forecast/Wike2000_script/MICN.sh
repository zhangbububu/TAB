python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"pred_len":24}' --model-name "time_series_library.MICN" --model-hyper-params '{"conv_kernel": [18, 12], "d_ff": 64, "d_model": 32, "dropout": 0.05, "lr": 0.001, "moving_avg": 25, "num_epochs": 15, "pred_len": 24, "seq_len": 36}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Wike2000/MICN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"pred_len":36}' --model-name "time_series_library.MICN" --model-hyper-params '{"conv_kernel": [18, 12], "d_ff": 256, "d_model": 128, "dropout": 0.05, "lr": 0.001, "moving_avg": 25, "num_epochs": 15, "pred_len": 36, "seq_len": 36}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Wike2000/MICN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"pred_len":48}' --model-name "time_series_library.MICN" --model-hyper-params '{"conv_kernel": [18, 12], "d_model": 512, "dropout": 0.05, "lr": 0.001, "moving_avg": 25, "num_epochs": 15, "pred_len": 48, "seq_len": 36, "d_ff": 2048}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Wike2000/MICN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"pred_len":60}' --model-name "time_series_library.MICN" --model-hyper-params '{"conv_kernel": [18, 12], "d_ff": 256, "d_model": 128, "dropout": 0.05, "lr": 0.001, "moving_avg": 25, "num_epochs": 15, "pred_len": 60, "seq_len": 36}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Wike2000/MICN"

