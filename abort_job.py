import os

command_str = 'curl -k --ntlm --user {}:{} "https://philly/api/abort?clusterId={}&jobId=application_{}"'.format(
                    'alias', 'passwd', 'rr1', '1569986296054_12605')

print(os.popen(command_str).readlines())