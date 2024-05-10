# UDP basic readout

## Introduction
The UDP scripts provided here is meant to start a board to a initialized state, and to set the board to a given configuration, start a readout, and stop the readout. This has no capabilities in capturing/collecting events and is meant to simply start a board's readout to send data over to a target machine that will handle the data collection.

## How to use

The python scripts, `init_board.py`, `start_capture.py`, and `stop_capture.py`, are provided to allow for board startup, starting a readout, and stopping a readout over UDP. The `run_scripts.bat` file contains an example values to initialize a board, start a capture and stop a capture.

### Init Board

This script initializes a given board to a default state.

**Parameters**:

- **model (-m, --model)**: model of your board
- **board IP (-b, --board_ip)**: the board's IP is set in the firmware file, contact us if you need a custom ip
- **board port (-bp, --board_port)**: firmware default is 4660
- **host IP (-host, --host_ip)**: the IP of the machine you are running the script on. If your machine has multiple network interfaces, be sure to set the IP to a network the board is able to communicate over.
- **host port (-hp, --host_port)**: Make sure to use an unused port


**Optional Parameters**:

- **config file (-cfg, --config_file)** the path to the config yaml if you plan on using a different startup configuration for your board
- **clock file (-clk, --clock_file)** the path to the clock file if you plan on using a custom clock configuration
- **debug (-d, --debug)** Flag to enable debug logging


### Start Readout

This script will set the board's parameters for readout, set the Board to send data to the target address, then starts the readout.


**Parameters**:

- **model (-m, --model)**: model of your board
- **board IP (-b, --board_ip)**: the board's IP is set in the firmware file, contact us if you need a custom ip
- **board port (-bp, --board_port)**: firmware default is 4660
- **host IP (-host, --host_ip)**: the IP of the machine you are running the script on. If your machine has multiple network interfaces, be sure to set the IP to a network the board is able to communicate over.
- **host port (-hp, --host_port)**: Make sure to use an unused port
- **target IP (-t, --target_ip)**: the IP of the machine you want the board to send data to. Ensure your board is on the same network as the target machine.
- **target port (-hp, --host_port)**: Make sure to use an unused port
- **readout window (-r, --readout_window)**: Readout window in the format: num windows, lookback, write after trigger
- **trigger mode (-trig, --trigger_mode)**: Trigger mode in which to initiate an event capture. Available options are: {imm, ext, self}. If trigger mode is set to self you arre required to set --trigger_values
- **lookback mode (-l, --lookback_mode)**: The lookback mode in which to record events available options are: {forced, trig}


**Optional/Conditional Parameters**:

- **trigger values (-trigval, --trigger_values)**: If the trigger mode is set to **self** trigger, trigger values is **required** to be set. Trigger values indicate at what value that each channel should trigger on. Trigger values are in the format: "val1 val2 val3 val4 ..."
- **dac values (-dac, --dac_values)**: The dac values to set for each channel. If not specified, will use the default dac values from the config file. DAC values are in the format: "val1 val2 val3 val4 ..."
- **debug (-d, --debug)** Flag to enable debug logging


### Stop Capture

This script will stop the Board's readout and set the Board to send data to the host IP.

**Parameters**:

- **model (-m, --model)**: model of your board
- **board IP (-b, --board_ip)**: the board's IP is set in the firmware file, contact us if you need a custom ip
- **board port (-bp, --board_port)**: firmware default is 4660
- **host IP (-host, --host_ip)**: the IP of the machine you are running the script on. If your machine has multiple network interfaces, be sure to set the IP to a network the board is able to communicate over.
- **host port (-hp, --host_port)**: Make sure to use an unused port


**Optional Parameters**:

- **debug (-d, --debug)** Flag to enable debug logging


## Readout Details

More details about the parameters that can be set in the start capture script.

### Readout Window

Readout window composes of 3 parameters, num windows, lookback, and write after trigger.

- **num windows**: the number of windows to readout per event
- **lookback**: the amount of events to lookback after receiving a trigger event
- **write after trigger**: the amount of windows to read after receiving a trigger event

To put theses parameters together, when the board receives a trigger event it will:

1. Continue to read %WRITE_AFTER_TRIGGER% windows
2. Lookback %LOOKBACK% windows
3. Digitize and return %NUM_WINDOWS% windows as an event


### Lookback Mode

This determines how the board returns data from its sampling array after receiving a trigger event.

- **Trigger Relative (trig)**: Will return data based on the description gave in readout window
- **Forced**: Will digitize %NUM_WINDOWS% worth of data starting from the %LOOKBACK% window of the sampling array, recommended only for debug usages.


### Trigger Mode

This determines how the board triggers an event to be readout.

- **Immediate (imm)**: Instantly digitizes data when possible to return as an event
- **External (ext)**: Triggers off of a sent Software trigger, or from the Hardware External Trigger input.
- **Self**: Triggers when the a channel receives an input that crosses the threshold set by %TRIGGER_VALUES%


### Trigger Values

This indicates the thresholds per channel that will indicate whether a self trigger has been activated. Be sure to specify a value per channel. To attain accurate trigger value to trigger on, refer to NaluDAQs trigger sweep to determine trigger values.

### DAC Values

This indicates the DAC value to set per DAC. Each channel usually has their own DAC to set. The DAC's range and resolution can be found in the board's yaml file, or could be found in python via `BOARD.params["ext_dac"]`.
