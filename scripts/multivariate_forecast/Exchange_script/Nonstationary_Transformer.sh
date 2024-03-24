python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"pred_len":96}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "p_hidden_dims": [256, 256], "p_hidden_layers": 2, "pred_len": 96, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/Nonstationary_Transformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"pred_len":192}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "p_hidden_dims": [64, 64, 64, 64], "p_hidden_layers": 4, "pred_len": 192, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/Nonstationary_Transformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"pred_len":336}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "p_hidden_dims": [256, 256], "p_hidden_layers": 2, "pred_len": 336, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/Nonstationary_Transformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"pred_len":720}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "p_hidden_dims": [256, 256], "p_hidden_layers": 2, "pred_len": 720, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/Nonstationary_Transformer"
