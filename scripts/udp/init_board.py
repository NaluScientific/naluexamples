#!/usr/bin/python3
import argparse
import logging
import sys

from helpers import _is_ip_valid, _is_port_valid
from naluconfigs import get_available_models
from naludaq.board import Board, startup_board
from naludaq.controllers import get_board_controller


def main():
    """Inits the board to a default state"""
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

    get_board_controller(brd).reset_board()

    if args.clock_file:
        brd.load_clockfile(args.clock_file)
    if args.config_file:
        brd.load_registers(args.config_file)

    startup_board(brd)
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
        "--config_file",
        "-cfg",
        type=str,
        required=False,
        help="Config file to set the board's registers to",
    )
    parser.add_argument(
        "--clock_file",
        "-clk",
        type=str,
        required=False,
        help="Clock file to program the board's clock",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
