python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Covid-19.csv" --strategy-args '{"pred_len":24}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.3, "e_layers": 3, "lr": 0.0025, "n_headers": 4, "num_epochs": 100, "patch_len": 24, "patience": 100, "pred_len": 24, "seq_len": 36, "stride": 2}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Covid-19/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Covid-19.csv" --strategy-args '{"pred_len":36}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.3, "e_layers": 3, "lr": 0.0025, "n_headers": 4, "num_epochs": 100, "patch_len": 24, "patience": 100, "pred_len": 36, "seq_len": 36, "stride": 2}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Covid-19/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Covid-19.csv" --strategy-args '{"pred_len":48}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.3, "e_layers": 3, "lr": 0.0025, "n_headers": 4, "num_epochs": 100, "patch_len": 24, "patience": 100, "pred_len": 48, "seq_len": 36, "stride": 2}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Covid-19/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Covid-19.csv" --strategy-args '{"pred_len":60}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.3, "e_layers": 3, "lr": 0.0025, "n_headers": 4, "num_epochs": 100, "patch_len": 24, "patience": 100, "pred_len": 60, "seq_len": 36, "stride": 2}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Covid-19/PatchTST"
