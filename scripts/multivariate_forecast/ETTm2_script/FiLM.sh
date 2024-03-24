python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"pred_len":96}' --model-name "time_series_library.FiLM" --model-hyper-params '{"dropout": 0.05, "lr": 0.001, "moving_avg": 25, "num_epochs": 30, "patience": 30, "pred_len": 96, "seq_len": 720, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTm2/FiLM"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"pred_len":192}' --model-name "time_series_library.FiLM" --model-hyper-params '{"dropout": 0.05, "lr": 0.001, "moving_avg": 25, "num_epochs": 30, "patience": 30, "pred_len": 192, "seq_len": 720, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTm2/FiLM"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"pred_len":336}' --model-name "time_series_library.FiLM" --model-hyper-params '{"factor": 3, "pred_len": 336, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTm2/FiLM"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"pred_len":720}' --model-name "time_series_library.FiLM" --model-hyper-params '{"batch_size": 16, "dropout": 0.05, "lr": 0.001, "moving_avg": 25, "num_epochs": 30, "patience": 30, "pred_len": 720, "seq_len": 2840, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTm2/FiLM"

