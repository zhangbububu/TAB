python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"pred_len":96}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"e_layers": 1, "factor": 3, "n_headers": 2, "pred_len": 96, "seq_len": 96, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh1/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"pred_len":192}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 512, "d_model": 128, "pred_len": 192, "seq_len": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh1/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"pred_len":336}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 512, "d_model": 128, "pred_len": 336, "seq_len": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh1/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"pred_len":720}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 128, "d_model": 32, "pred_len": 720, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh1/PatchTST"

