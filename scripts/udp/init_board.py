import argparse
import re
import logging
import sys

from naluconfigs import get_available_models
from naludaq.board import Board, startup_board
from naludaq.controllers import get_board_controller

def main():
    """Inits the board to a default state"""
    args = parse_args(sys.argv[1:])

    if not _is_ipstr_valid(args.board_ip):
        raise ValueError("Invalid format: Board IP")
    if not _is_ipstr_valid(args.host_ip):
        raise ValueError("Invalid format: Host IP")
    if args.debug:
        setup_logger()

    BOARD_MODEL = args.model
    BOARD_IP = _parse_ip_str(args.board_ip)
    HOST_IP = _parse_ip_str(args.host_ip)

    BOARD = Board(BOARD_MODEL)
    BOARD.get_udp_connection(BOARD_IP, HOST_IP)

    BC = get_board_controller(BOARD)
    BC.reset_board()

    startup_board(BOARD)
    BOARD.disconnect()

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
        description="Inits a board to a default state"
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
        help="Board IP in the format ADDRESS:PORT",
    )
    parser.add_argument(
        "--host_ip",
        "-host",
        type=str,
        required=True,
        help="IP of the host computer running the script in the format ADDRESS:PORT",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action='store_true',
    )
    return parser.parse_args(argv)

def _parse_ip_str(ip: str) -> tuple:
    splitted = ip.split(":")
    return (splitted[0], int(splitted[1]))


def _is_ipstr_valid(ip: str):
    ip_port_regex = r"^((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d):([1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5])$"
    if re.match(ip_port_regex, ip):
        return True
    return False

if __name__ == "__main__":
    main()
