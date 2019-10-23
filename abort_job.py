import os

command_str = 'curl -k --ntlm --user {}:{} "https://philly/api/abort?clusterId={}&jobId=application_{}"'.format(
                    'v-miyin', '4QTGGG6p-', 'rr1', '1569986296054_12496')

print(os.popen(command_str).readlines())