# localize system command calls
import os
import subprocess
import time
import logging
import platform


def ping(host):
    """
    Returns True if host responds to a ping request

    Compatible on both windows and linux

    """
    import platform

    # Ping parameters as function of OS
    ping_arg = "-n" if platform.system().lower() == "windows" else "-c"

    # Block for ping result
    return subprocess.run(['ping', ping_arg, '1', host]).returncode


def restart_myo():
    # Use this function to issue a restart command on the myo services.
    # Note this gets trickier when primary/back myo bands are used.
    # first check if the service is active, only then issue restart
    #
    # Function is only for use with Linux

    if not platform.system().lower() == 'linux':
        return
    else:
        logging.critical('Restarting MYO')

    if not subprocess.run(['systemctl', 'is-enabled', 'mpl_myo1.service']).returncode:
        subprocess.run(['sudo', 'systemctl', 'stop', 'mpl_myo1.service'])
        subprocess.run(['sleep', '3'])
        subprocess.run(['sudo', 'systemctl', 'start', 'mpl_myo1.service'])

    if not subprocess.run(['systemctl', 'is-enabled', 'mpl_myo2.service']).returncode:
        subprocess.run(['sudo', 'systemctl', 'stop', 'mpl_myo2.service'])
        subprocess.run(['sleep', '3'])
        subprocess.run(['sudo', 'systemctl', 'start', 'mpl_myo2.service'])


def change_myo(val):
    # Use this command to change the active pair of myos searched during startup
    # This is accomplished by stopping/disabling and enabling/starting the respective services
    # Set one is mpl_myo1
    # Set two is mpl_myo2
    #
    # Function is only for use with Linux

    if not platform.system().lower() == 'linux':
        return
    else:
        logging.critical('Changing to MYO set: ' + str(val))

    if val == 1:
        subprocess.run(['sudo', 'systemctl', 'stop', 'mpl_myo2.service'])
        subprocess.run(['sudo', 'systemctl', 'disable', 'mpl_myo2.service'])
        subprocess.run(['sudo', 'systemctl', 'enable', 'mpl_myo1.service'])
        subprocess.run(['sudo', 'systemctl', 'start', 'mpl_myo1.service'])
    elif val == 2:
        subprocess.run(['sudo', 'systemctl', 'stop', 'mpl_myo1.service'])
        subprocess.run(['sudo', 'systemctl', 'disable', 'mpl_myo1.service'])
        subprocess.run(['sudo', 'systemctl', 'enable', 'mpl_myo2.service'])
        subprocess.run(['sudo', 'systemctl', 'start', 'mpl_myo2.service'])


def shutdown():
    # os.system("sudo shutdown -h now")
    # TODO: This isn't a great strategy for issuing low battery warning, but it works
    import socket
    import time
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        # Issue vibrate command for myo listening on this port
        s.sendto(bytearray([0, 1]), ('localhost', 16001))
        time.sleep(0.5)
        s.sendto(bytearray([0, 1]), ('localhost', 16001))
        time.sleep(0.5)
        s.sendto(bytearray([0, 1]), ('localhost', 16001))
        time.sleep(5)


def set_system_time(date_num, time_error=30.0):
    # Check time compared to system time.  If out of limits, then set system time
    #
    # Parameters
    # ----------
    # date_num : float
    #   Numeric value corresponding to the time in seconds since Jan 1, 1970, 00:00:00.000 GMT
    #
    # time_error :  float
    #   Max allowable error before resetting time (seconds)

    # get system time (convert to milliseconds)
    system_time = time.time()
    user_time = date_num

    system_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(system_time))
    user_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(user_time))

    e_time = system_time - user_time

    logging.critical('UTC User time number: ' + str(user_time))
    logging.critical('UTC System time number: ' + str(system_time))
    logging.critical('UTC User time: ' + user_time_str)
    logging.critical('UTC System time: ' + system_time_str)
    logging.critical('Time shift: ' + str(e_time))

    if abs(e_time) > time_error:
        logging.critical('Time Error is out of bounds.  Resetting time: ' + str(e_time))
        # timedatectl set-time '16:10:40 2015-11-20'
        send_system_command(['sudo', 'timedatectl', 'set-ntp', '0'])
        send_system_command(['sudo', 'timedatectl', 'set-time', user_time_str])
        send_system_command(['sudo', 'hwclock', '--systohc'])
        send_system_command(['sudo', 'hwclock', '--systohc'])


def send_system_command(cmd):
    # Route all os.system calls through here to allow system checks and ensure these are enabled
    #
    # This function creates the subprocess immediately and is non-blocking

    logging.critical('SysCmd: ' + " ".join(cmd))

    if platform.system() == 'Linux':
        # os.system(cmd)
        # calling without .wait() option to ensure system responds promptly
        subprocess.Popen(cmd)
        return
    else:
        logging.critical('System command not permitted on this platform')
        return None
