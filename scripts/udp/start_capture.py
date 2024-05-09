#!/usr/bin/python3
import argparse
import logging
import sys

from helpers import _is_ipstr_valid, _parse_ip_str
from naluconfigs import get_available_models
from naludaq.board import Board
from naludaq.controllers import get_board_controller, get_readout_controller


def main():
    """Starts data capture for a board"""
    args = parse_args(sys.argv[1:])
    if args.debug:
        setup_logger()

    board_model = args.model

    if not _is_ipstr_valid(args.board_ip):
        raise ValueError("Invalid format: Board IP")
    board_ip = _parse_ip_str(args.board_ip)

    if args.host_ip is None:
        host_ip = ("127.0.0.1", 4660)
    else:
        if not _is_ipstr_valid(args.host_ip):
            raise ValueError("Invalid format: Host IP")
        host_ip = _parse_ip_str(args.host_ip)

    atof = Board(board_model)
    atof.get_udp_connection(board_ip, host_ip)

    get_readout_controller(atof).set_read_window(*args.read_window)

    get_board_controller(atof).start_readout(
        trig=args.trigger_mode,
        lb=args.lookback_mode,
    )
    atof.disconnect()


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
        help="Board IP in the format ADDRESS:PORT",
    )
    parser.add_argument(
        "--host_ip",
        "-host",
        type=str,
        required=True,
        help="IP of the host computer running the script in the format ADDRESS:PORT, Defaults to first local ip found with port 4660",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
    )

    parser.add_argument(
        "--readout_window",
        nargs=3,
        type=int,
        help="Read Window in the format: num windows, lookback, write after trigger",
    )

    parser.add_argument(
        "--trigger_mode",
        "-t",
        type=str,
        required=True,
        help="ext: External trigger, using the trig_in on the board or software commands\nimm: Immediate trigger will trigger automatically without signal\nself: Self trigger will trigger on analog signals.\n",
        choices=["imm", "ext", "self"],
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
