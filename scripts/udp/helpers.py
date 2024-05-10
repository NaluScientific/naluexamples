import ipaddress


def _parse_ip_str(ip: str) -> tuple:
    splitted = ip.split(":")
    return (splitted[0], int(splitted[1]))


def _is_ip_valid(ip: str):
    """Will check if the IP string is valid, will not check if 'ip' is a string."""
    try:
        ipaddress.ip_address(ip)
    except (ValueError, SyntaxError):
        return False
    return True

def _is_port_valid(port: str):
    """Checks if port is a positive integer <= 65535"""
    try:
        port_int = int(port)
    except ValueError:
        return False
    if port_int <= 0 or port_int > 65535:
        return False
    return True
