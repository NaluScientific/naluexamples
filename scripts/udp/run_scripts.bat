set "BOARD_IP=192.168.1.68:4660"
set "HOST_IP=192.168.1.165:4660"
set "MODEL=aardvarcv3"
set "READ_WINDOW=16 16 16"
set "TRIGGER_MODE=ext"
set "LOOKBACK_MODE=trig"

@REM python init_board.py -m %MODEL% -b %BOARD_IP% -host %HOST_IP%
python start_capture.py -m %MODEL% -b %BOARD_IP% -host %HOST_IP% --read_window %READ_WINDOW% -t %TRIGGER_MODE% -l %LOOKBACK_MODE%
python stop_capture.py -m %MODEL% -b %BOARD_IP% -host %HOST_IP%