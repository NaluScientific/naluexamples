set "BOARD_IP=192.168.1.68"
set "BOARD_PORT=4660"
set "HOST_IP=192.168.1.165"
set "HOST_PORT=4660"
set "TARGET_IP=192.168.1.165"
set "TARGET_PORT=4660"
set "MODEL=aardvarcv3"
set "READOUT_WINDOW=16 16 16"
set "TRIGGER_MODE=ext"
set "LOOKBACK_MODE=trig"

python init_board.py -m %MODEL% -b %BOARD_IP% -bp %BOARD_PORT% -host %HOST_IP% -hp %HOST_PORT% 
python start_capture.py -m %MODEL% -b %BOARD_IP% -bp %BOARD_PORT% -host %HOST_IP% -hp %HOST_PORT% -t %TARGET_IP% -tp %TARGET_PORT% --readout_window %READOUT_WINDOW% -trig %TRIGGER_MODE% -l %LOOKBACK_MODE%
python stop_capture.py -m %MODEL% -b %BOARD_IP% -bp %BOARD_PORT% -host %HOST_IP% -hp %HOST_PORT% 
