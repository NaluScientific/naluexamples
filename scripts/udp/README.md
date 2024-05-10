# UDP basic readout

## Introduction
The UDP scripts provided here is meant to start a board to a initialized state, and to set the board to a given configuration, start a readout, and stop the readout. This has no capabilities in capturing/collecting events and is meant to simply start a board's readout to send data over to a target machine that will handle the data collection.

## How to use

The python scripts, `init_board.py`, `start_capture.py`, and `stop_capture.py`, are provided to allow for board startup, starting a readout, and stopping a readout over UDP. The `run_scripts.bat` file contains an example values to initialize a board, start a capture and stop a capture.
### Init Board
This script initializes a given board to a default state.<br>
<b>Parameters</b>:
<ul> 
    <li> <b>model (-m, --model)</b>: model of your board </li>
    <li> <b>board IP (-b, --board_ip)</b>: the board's IP is set in the firmware file, contact us if you need a custom ip
    <li> <b>board port (-bp, --board_port)</b>: firmware default is 4660</li>
    <li> <b>host IP (-host, --host_ip)</b>: the IP of the machine you are running the script on. If your machine has multiple network interfaces, be sure to set the IP to a network the board is able to communicate over.
    <li> <b>host port (-hp, --host_port)</b>: Make sure to use an unused port</li>
</ul>

<b>Optional Parameters</b>:
<ul> 
    <li> <b>config file (-cfg, --config_file)</b> the path to the config yaml if you plan on using a different startup configuration for your board</li>
    <li> <b>clock file (-clk, --clock_file)</b> the path to the clock file if you plan on using a custom clock configuration</li>
    <li> <b>debug (-d, --debug)</b> Flag to enable debug logging</li>
</ul>

### Start Readout
This script will set the board's parameters for readout, set the Board to send data to the target address, then starts the readout.<br> 
<b>Parameters</b>:
<ul> 
    <li> <b>model (-m, --model)</b>: model of your board </li>
    <li> <b>board IP (-b, --board_ip)</b>: the board's IP is set in the firmware file, contact us if you need a custom ip
    <li> <b>board port (-bp, --board_port)</b>: firmware default is 4660</li>
    <li> <b>host IP (-host, --host_ip)</b>: the IP of the machine you are running the script on. If your machine has multiple network interfaces, be sure to set the IP to a network the board is able to communicate over.
    <li> <b>host port (-hp, --host_port)</b>: Make sure to use an unused port</li>
    <li> <b>target IP (-t, --target_ip)</b>: the IP of the machine you want the board to send data to. Ensure your board is on the same network as the target machine.
    <li> <b>target port (-hp, --host_port)</b>: Make sure to use an unused port</li>
    <li><b>readout window (-r, --readout_window)</b>: Readout window in the format: num windows, lookback, write after trigger</li>
    <li><b>trigger mode (-trig, --trigger_mode)</b>: Trigger mode in which to initiate an event capture. Available options are: {imm, ext, self}. If trigger mode is set to self you arre required to set --trigger_values</li>
    <li><b>lookback mode (-l, --lookback_mode)</b>: The lookback mode in which to record events available options are: {forced, trig}</li>
</ul>
<b>Optional/Conditional Parameters</b>:
<ul> 
    <li><b>trigger values (-trigval, --trigger_values)</b>: If the trigger mode is set to <b>self</b> trigger, trigger values is <b>required</b> to be set. Trigger values indicate at what value that each channel should trigger on. Trigger values are in the format: "val1 val2 val3 val4 ..."
    <li> <b>dac values (-dac, --dac_values)</b>: The dac values to set for each channel. If not specified, will use the default dac values from the config file. DAC values are in the format: "val1 val2 val3 val4 ..."</li>
    <li> <b>debug (-d, --debug)</b> Flag to enable debug logging</li>
</ul>

### Stop Capture
This script will stop the Board's readout and set the Board to send data to the host IP.<br>
<b>Parameters</b>:
<ul> 
    <li> <b>model (-m, --model)</b>: model of your board </li>
    <li> <b>board IP (-b, --board_ip)</b>: the board's IP is set in the firmware file, contact us if you need a custom ip
    <li> <b>board port (-bp, --board_port)</b>: firmware default is 4660</li>
    <li> <b>host IP (-host, --host_ip)</b>: the IP of the machine you are running the script on. If your machine has multiple network interfaces, be sure to set the IP to a network the board is able to communicate over.
    <li> <b>host port (-hp, --host_port)</b>: Make sure to use an unused port</li>
</ul>

<b>Optional Parameters</b>:
<ul> 
    <li> <b>debug (-d, --debug)</b> Flag to enable debug logging</li>
</ul>

## Readout Details

Explain how and what the different variables do.

### Readout Window
Readout window composes of 3 parameters, num windows, lookback, and write after trigger.
<ul> 
    <li> <b>num windows</b>: the number of windows to readout per event </li>
    <li> <b>lookback</b>: the amount of events to lookback after receiving a trigger event </li>
    <li> <b>write after trigger</b>: the amount of windows to read after receiving a trigger event </li>
</ul> 

To put theses parameters together, when the board receives a trigger event it will:
<ol>
    <li> Continue to read %WRITE_AFTER_TRIGGER% windows </li>
    <li> Lookback %LOOKBACK% windows </li>
    <li> Digitize and return %NUM_WINDOWS% windows as an event</li>
</ol>

### Lookback Mode
This determines how the board returns data from its sampling array after receiving a trigger event.

<ul> 
    <li> <b>Trigger Relative (trig)</b>: Will return data based on the description gave in readout window</li>
    <li> <b>Forced</b>: Will digitize %NUM_WINDOWS% worth of data starting from the %LOOKBACK% window of the sampling array, recommended only for debug usages.</li>
</ul> 

### Trigger Mode
This determines how the board triggers an event to be readout.
<ul> 
    <li> <b>Immediate (imm)</b>: Instantly digitizes data when possible to return as an event</li>
    <li> <b>External (ext)</b>: Triggers off of a sent Software trigger, or from the Hardware External Trigger input.</li>
    <li> <b>Self</b>: Triggers when the a channel receives an input that crosses the threshold set by %TRIGGER_VALUES% </li>
</ul> 

### Trigger Values
This indicates the thresholds per channel that will indicate whether a self trigger has been activated. Be sure to specify a value per channel. To attain accurate trigger value to trigger on, refer to NaluDAQs trigger sweep to determine trigger values.

### DAC Values
This indicates the DAC value to set per DAC. Each channel usually has their own DAC to set. The DAC's range and resolution can be found in the board's yaml file, or could be found in python via `BOARD.params["ext_dac"]`.
