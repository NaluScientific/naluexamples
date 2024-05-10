#!/usr/bin/python3
import argparse
import logging
import sys

from helpers import _is_ip_valid, _is_port_valid
from naluconfigs import get_available_models
from naludaq.board import Board
from naludaq.controllers import get_board_controller


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

    brd = Board(board_model)
    brd.get_udp_connection(board_ip, host_ip)

    get_board_controller(brd).stop_readout()

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
    parser = argparse.ArgumentParser(
        description="Stops the board capture, and sets the return address to the host's address"
    )
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
        "--debug",
        "-d",
        action="store_true",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
