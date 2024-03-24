python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"pred_len":96}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 8, "lr": 0.005, "pred_len": 96, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"pred_len":192}' --model-name "time_series_library.DLinear" --model-hyper-params '{"factor": 3, "pred_len": 192, "seq_len": 96, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"pred_len":336}' --model-name "time_series_library.DLinear" --model-hyper-params '{"factor": 3, "pred_len": 336, "seq_len": 96, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"pred_len":720}' --model-name "time_series_library.DLinear" --model-hyper-params '{"d_ff": 64, "d_model": 32, "pred_len": 720, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/DLinear"

