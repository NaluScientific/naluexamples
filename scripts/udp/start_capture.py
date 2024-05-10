#!/usr/bin/python3
import argparse
import logging
import sys

from helpers import _is_ip_valid, _is_port_valid
from naluconfigs import get_available_models
from naludaq.board import Board
from naludaq.controllers import (
    get_board_controller,
    get_connection_controller,
    get_dac_controller,
    get_readout_controller,
    get_trigger_controller,
)


def main():
    """Starts data capture for a board"""
    args = parse_args(sys.argv[1:])
    if args.debug:
        setup_logger()

    board_model = args.model

    if not _is_ip_valid(args.board_ip) or not _is_port_valid(args.board_port):
        raise ValueError("Invalid format: Board IP")
    board_ip = (args.board_ip, int(args.board_port))

    if not _is_ip_valid(args.host_ip) or not _is_port_valid(args.board_port):
        raise ValueError("Invalid format: Host IP")
    host_ip = (args.host_ip, int(args.host_port))

    if not _is_ip_valid(args.target_ip) or not _is_port_valid(args.target_port):
        raise ValueError("Invalid format: Target IP")
    target_ip = (args.target_ip, int(args.target_port))

    if args.trigger_mode == "self" and not args.trigger_values:
        raise ValueError(
            "Trigger mode is self, please provide trigger values for channels"
        )

    brd = Board(board_model)
    brd.get_udp_connection(board_ip, host_ip)

    if args.trigger_values:
        brd.trigger_values = args.trigger_values
        get_trigger_controller(brd).write_triggers()
    if args.dac_values:
        for chan, val in enumerate(args.dac_values):
            get_dac_controller(brd).set_single_dac(chan, val)
    get_readout_controller(brd).set_read_window(*args.readout_window)

    # Set receiver address to the target computer's address
    brd.connection_info["receiver_addr"] = target_ip
    get_connection_controller(brd)._configure_ethernet()

    get_board_controller(brd).start_readout(
        trig=args.trigger_mode,
        lb=args.lookback_mode,
    )
    brd.disconnect()


def setup_logger(level=logging.DEBUG):
    """Setup a basic logger."""
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)-30s [%(levelname)-6s]: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    suppress = [
        "naludaq.UART",
        "naludaq.FTDI",
    ]
    for name in suppress:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    return logger


def parse_args(argv):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Inits a board to a default state")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Board model",
        required=True,
        choices=get_available_models(),
    )
    parser.add_argument(
        "--board_ip",
        "-b",
        type=str,
        required=True,
        help="Board IP in the IPv4 format",
    )
    parser.add_argument(
        "--board_port",
        "-bp",
        type=str,
        required=True,
        help="Board port",
    )
    parser.add_argument(
        "--host_ip",
        "-host",
        type=str,
        required=True,
        help="IP of the host computer running the script in the IPv4 format",
    )
    parser.add_argument(
        "--host_port",
        "-hp",
        type=str,
        required=True,
        help="Host port",
    )
    parser.add_argument(
        "--target_ip",
        "-t",
        type=str,
        required=True,
        help="IP of the target computer receiving the data in IPv4 format",
    )
    parser.add_argument(
        "--target_port",
        "-tp",
        type=str,
        required=True,
        help="Target port",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
    )
    parser.add_argument(
        "--dac_values",
        "-dac",
        type=int,
        required=False,
        help="DAC values to set the board's channels to",
        nargs="+",
    )
    parser.add_argument(
        "--readout_window",
        "-r",
        nargs=3,
        type=int,
        help="Read Window in the format: num windows, lookback, write after trigger",
    )

    parser.add_argument(
        "--trigger_mode",
        "-trig",
        type=str,
        required=True,
        help="ext: External trigger, using the trig_in on the board or software commands\nimm: Immediate trigger will trigger automatically without signal\nself: Self trigger will trigger on analog signals.\n",
        choices=["imm", "ext", "self"],
    )
    parser.add_argument(
        "--trigger_values",
        "-trigval",
        type=int,
        required=False,
        help="Trigger values to set per channel in the format: val1 val2 val3 ...",
        nargs="+",
    )
    parser.add_argument(
        "--lookback_mode",
        "-l",
        type=str,
        required=False,
        help="",
        choices=["forced", "trig"],
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
