import os
import six


def ping(host):
    """
    Returns True if host responds to a ping request
    """
    import platform

    # Ping parameters as function of OS
    ping_str = "-n 1" if platform.system().lower() == "windows" else "-c 1"

    # Ping
    return os.system("ping " + ping_str + " " + host) == 0


def get_address(url):
    # convert address url string to get hostname and port as tuple for socket interface
    # error checking is centralized here
    #
    # E.g. //127.0.0.1:1234 becomes:
    #   hostname = 127.0.0.1
    #   port = 1234
    a = six.moves.urllib.parse.urlparse(url)

    assert isinstance(a.hostname, six.string_types), "hostname is not a string: %r" % a.hostname
    assert isinstance(a.port, six.integer_types), "port is not an integer: %r" % a.port
    return a.hostname, a.port


def restart_myo(unused):
    # Use this function to issue a restart command on the myo services.
    # Note this gets trickier when primary/back myo bands are used.
    # first check if the service is active, only then issue restart
    os.system("sudo systemctl is-enabled mpl_myo1.service | grep 'enabled' > /dev/null && sudo systemctl stop mpl_myo1.service && sleep 3 && sudo systemctl start mpl_myo1.service")
    os.system("sudo systemctl is-enabled mpl_myo2.service | grep 'enabled' > /dev/null && sudo systemctl stop mpl_myo2.service && sleep 3 && sudo systemctl start mpl_myo2.service")


def change_myo(val):
    # Use this command to change the active pair of myos searched during startup
    # This is accomplished by stopping/disabling and enabling/starting the respective services
    # Set one is mpl_myo1
    # Set two is mpl_myo2

    if val == 1:
        os.system("sudo systemctl stop mpl_myo2.service")
        os.system("sudo systemctl disable mpl_myo2.service")
        os.system("sudo systemctl enable mpl_myo1.service")
        os.system("sudo systemctl start mpl_myo1.service")
    elif val == 2:
        os.system("sudo systemctl stop mpl_myo1.service")
        os.system("sudo systemctl disable mpl_myo1.service")
        os.system("sudo systemctl enable mpl_myo2.service")
        os.system("sudo systemctl start mpl_myo2.service")


def reboot():
    os.system("sudo shutdown -r now")


def shutdown():
    # os.system("sudo shutdown -h now")
    import socket
    import time
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        #
        s.sendto(bytearray([1]), ('localhost', 16001))
        time.sleep(0.5)
        s.sendto(bytearray([1]), ('localhost', 16001))
        time.sleep(0.5)
        s.sendto(bytearray([1]), ('localhost', 16001))
        time.sleep(5)
