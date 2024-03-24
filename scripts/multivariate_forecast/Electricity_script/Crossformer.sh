python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"pred_len":96}' --model-name "time_series_library.Crossformer" --model-hyper-params '{"d_ff": 128, "d_model": 64, "dropout": 0.2, "e_layers": 3, "factor": 10, "lr": 0.0005, "n_headers": 2, "num_epochs": 20, "pred_len": 96, "seg_len": 12, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/Crossformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"pred_len":192}' --model-name "time_series_library.Crossformer" --model-hyper-params '{"d_ff": 512, "d_model": 256, "pred_len": 192, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/Crossformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"pred_len":336}' --model-name "time_series_library.Crossformer" --model-hyper-params '{"d_ff": 512, "d_model": 256, "pred_len": 336, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/Crossformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"pred_len":720}' --model-name "time_series_library.Crossformer" --model-hyper-params '{"d_ff": 512, "d_model": 256, "pred_len": 720, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/Crossformer"

