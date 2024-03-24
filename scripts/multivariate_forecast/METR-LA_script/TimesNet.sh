python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "METR-LA.csv" --strategy-args '{"pred_len":96}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 512, "d_model": 512, "factor": 3, "pred_len": 96, "seq_len": 96, "top_k": 5}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "METR-LA/TimesNet"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "METR-LA.csv" --strategy-args '{"pred_len":192}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 512, "d_model": 512, "factor": 3, "pred_len": 192, "seq_len": 96, "top_k": 5}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "METR-LA/TimesNet"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "METR-LA.csv" --strategy-args '{"pred_len":336}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 512, "d_model": 512, "factor": 3, "pred_len": 336, "seq_len": 96, "top_k": 5}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "METR-LA/TimesNet"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "METR-LA.csv" --strategy-args '{"pred_len":720}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 512, "d_model": 256, "pred_len": 720, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "METR-LA/TimesNet"

