import argparse
import re
import logging
import socket
import sys

from naluconfigs import get_available_models
from naludaq.board import Board, startup_board
from naludaq.controllers import get_board_controller


def main():
    """Inits the board to a default state"""
    args = parse_args(sys.argv[1:])
    if args.debug:
        setup_logger()

    board_model = args.model

    if not _is_ipstr_valid(args.board_ip):
        raise ValueError("Invalid format: Board IP")
    board_ip = _parse_ip_str(args.board_ip)

    if args.host_ip is None:
        local_hostname = socket.gethostname()
        ip_addresses = socket.gethostbyname_ex(local_hostname)[2]
        filtered_ips = [ip for ip in ip_addresses if not ip.startswith("127.")]
        host_ip = (filtered_ips[0], 4660)
    else:
        if not _is_ipstr_valid(args.host_ip):
            raise ValueError("Invalid format: Host IP")
        host_ip = _parse_ip_str(args.host_ip)

    atof = Board(board_model)
    atof.get_udp_connection(board_ip, host_ip)

    get_board_controller(atof).reset_board()

    startup_board(atof)
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
        required=False,
        help="IP of the host computer running the script in the format ADDRESS:PORT, Defaults to first local ip found with port 4660",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
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
