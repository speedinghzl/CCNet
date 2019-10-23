import argparse
import os
import time
def ompi_rank():
    """Find OMPI world rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)


def ompi_size():
    """Find OMPI world size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)
print('rank: {}'.format(ompi_rank()))
print('env:')
print(os.environ)
os.system("ls")
parser = argparse.ArgumentParser(description='Helper run')
# general
parser.add_argument('--auth', help='auth', required=True, type=str)
parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
parser.add_argument('--branch', help="branch of code", type=str, default='pytorch-1.1_philly')
args, rest = parser.parse_known_args()
print('----------- run_on_philly info ----------')
print(args)
print(rest)

# is_worker = ompi_rank() != 0 and ompi_size() > 1
os.system("git clone https://{0}@github.com/zdaxie/CCNet -b {1} $HOME/CCNet".format(args.auth, args.branch))
# only master need to install package
os.chdir(os.path.expanduser('~/CCNet'))
# os.system('ln -s /hdfs/resrchprojvc4/zhez/data .')
# os.system('ln -s /hdfs/resrchprojvc4/zhez/model .')
# os.system('ln -s /home/zhez/data_local .')
os.system('ls')
os.system('bash init.sh')
os.system('bash compile.sh')
os.chdir(os.path.expanduser('~/CCNet'))
print('Rank {}'.format(ompi_rank()))
os.system("experiments/{}".format(args.cfg))