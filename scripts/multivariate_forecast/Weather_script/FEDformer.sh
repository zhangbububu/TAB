python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"pred_len":96}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "pred_len": 96, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Weather/FEDformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"pred_len":192}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"batch_size": 8, "d_ff": 512, "d_model": 256, "dropout": 0.05, "factor": 3, "moving_avg": 25, "pred_len": 192, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Weather/FEDformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"pred_len":336}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "pred_len": 336, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Weather/FEDformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"pred_len":720}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "pred_len": 720, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Weather/FEDformer"
