import ipaddress


def _parse_ip_str(ip: str) -> tuple:
    splitted = ip.split(":")
    return (splitted[0], int(splitted[1]))


def _is_ipstr_valid(ip: str):
    """Will check if the IP string is valid, will not check if 'ip' is a string."""
    try:
        ipaddress.ip_address(ip)
    except (ValueError, SyntaxError):
        return False
    return True
