python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"pred_len":96}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 64, "d_model": 32, "pred_len": 96, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTm2/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"pred_len":192}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 64, "d_model": 32, "pred_len": 192, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTm2/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"pred_len":336}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 64, "d_model": 32, "pred_len": 336, "seq_len": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTm2/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"pred_len":720}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 64, "d_model": 32, "pred_len": 720, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTm2/PatchTST"

