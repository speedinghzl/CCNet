import requests
from requests_ntlm import HttpNtlmAuth
import argparse
import os
import pprint
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def parse_args():
    parser = argparse.ArgumentParser(description='hack philly vc')
    parser.add_argument(
        '--name', type=str, help='Job Name', default='test')
    parser.add_argument(
        '--user', type=str, help='hack user name', default='v-miyin')
    parser.add_argument('--VcId', type=str, help='VcId', default='resrchprojvc15')
    parser.add_argument(
        '--ClusterId', type=str, help='ClusterId', default='rr1')
    parser.add_argument('--gpus', type=int, help='Gpu numbers', default=4)
    parser.add_argument(
        '--branch', type=str, help='git branch to pull', default='pytorch-1.1_philly')

    return parser.parse_known_args()


args, rest = parse_args()
extra_args = ' '.join(rest)

submitted_jobId = []
cfgs = [
    'Philly_Test.sh',
]
pprint.pprint(cfgs)

gpus8_clusters = ['sc3', 'philly-prod-cy4']

submit_url = 'https://philly/api/v2/submit'
submit_headers = {'Content-Type': 'application/json'}
pwd = {'{alias}': '{passwd}'}

ClusterId = args.ClusterId
VcId = args.VcId
name = args.name + '_' + os.path.splitext(os.path.basename(cfgs[0]))[0]
user = args.user
branch = args.branch
auth = '{git cridential}'
gpus = args.gpus
philly_auth = HttpNtlmAuth(user, pwd[user])
submit_data = {}
submit_data["ClusterId"] = ClusterId
submit_data["VcId"] = VcId
submit_data["JobName"] = name
submit_data["UserName"] = user
submit_data["BuildId"] = 0
submit_data["ToolType"] = None
submit_data["Inputs"] = [
    {
        "Name": "dataDir",
        "Path": "/hdfs/resrchprojvc15/zhez/" #"Path": "/hdfs/nextmsra/{alias}/"
    },
]
submit_data["Outputs"] = []
submit_data["IsDebug"] = True
submit_data["RackId"] = "anyConnected"
submit_data["MinGPUs"] = gpus
submit_data["PrevModelPath"] = None
submit_data["ExtraParams"] = "--cfg {0} --branch {1} --auth {2} {3}"
submit_data["SubmitCode"] = "p"
submit_data["IsMemCheck"] = False
submit_data["IsCrossRack"] = False
submit_data["Registry"] = "phillyregistry.azurecr.io"
submit_data["Repository"] = "philly/jobs/custom/pytorch" # use Han Xue's Docker "philly/jobs/custom/pytorch"
submit_data["Tag"] = "pytorch1.1.0-tensorflow1.7.0-py36-cuda9-mpi-nccl-hvd-apex-video" # define use which docker
submit_data["OneProcessPerContainer"] = False
submit_data["NumOfContainers"] = str(
    gpus // 8) if ClusterId in gpus8_clusters else str(gpus // 4)
submit_data["dynamicContainerSize"] = False

submit_data["volumes"] = {
        "myblob": {
                "type": "blobfuseVolume",
                "storageAccount": "vcwestus2",
                "containerName": "data",
                "path": "/blob/data"
            }
}

submit_data["credentials"] = {
        "storageAccounts": {
            "vcwestus2": {
                "key": "{keys found in azure}"
            }
        }
    }


if 'resrchprojvc' not in VcId:
    submit_data["Queue"] = "bonus"
for cfg in cfgs:

    data = submit_data.copy()
    data["ConfigFile"] = "/hdfs/resrchprojvc15/zhez/run_on_philly_pytorch_seg.py" # data["ConfigFile"] = "/hdfs/nextmsra/{alias}/run_mmdet_on_philly_dist.py"
    if ClusterId in ['gcr', 'rr1', 'rr2', 'cam', 'philly-prod-cy4']:
        data[
            "CustomMPIArgs"] = "env CUDA_CACHE_DISABLE=1 NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=INFO OMP_NUM_THREADS=2"
    else:
        data[
            "CustomMPIArgs"] = "env CUDA_CACHE_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO OMP_NUM_THREADS=2"

    data['ExtraParams'] = data['ExtraParams'].format(cfg, branch, auth,
                                                     extra_args)

    r = requests.post(
        json=data,
        url=submit_url,
        auth=philly_auth,
        headers=submit_headers,
        verify=False)
    if r.status_code == 200:
        res = r.json()
        print("submit {0} to {1} successfully".format(data['JobName'],
                                                      ClusterId))
        print(res)
        res_dict = {}
        res_dict["JobName"] = data["JobName"]
        res_dict["AppId"] = res['jobId']
        res_dict["cfg"] = cfg
        res_dict["Link"] = "https://philly/#/job/{}/{}/{}".format(
            ClusterId, VcId, res['jobId'][12:])
        submitted_jobId.append(res_dict)
    else:
        print(r)
        print('submit failed with status_code {}'.format(r.status_code))

pprint.pprint(submitted_jobId)
