import argparse
import os
import sys
import subprocess
import yaml


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


PROJ_NAME = os.path.abspath(__file__).split(os.path.sep)[-3]
print(f"PROJ_NAME = {PROJ_NAME}")

parser = argparse.ArgumentParser("onenode")
# parameters from start command
parser.add_argument('--init_method', type=str, default=None,
                    help='master address')
parser.add_argument('--world_size', type=int, default=1,
                    help='Total number of gpus')

# parameters for this
parser.add_argument('--task_index', type=str, default="-1",
                    help='train job index')
parser.add_argument('--overrides_file', type=str, default="",
                    help='overrides for hydra')
parser.add_argument('--overrides', type=str, default="",
                    help='overrides for hydra')
parser.add_argument('--cloud', type=str2bool, default=False, nargs='?',
                    const=True, help='whether setup env')
parser.add_argument('--try_run', type=str2bool, default=False, nargs='?',
                    const=True, help='whether run whole data')

args, unparsed = parser.parse_known_args()
print(f"got args: {args} and unparsed {unparsed}")

# construct overrides
overrides = args.overrides.strip("\"")
if not overrides.endswith(" ") and overrides != "":
    overrides += " "
if os.path.isfile(args.overrides_file):
    with open(args.overrides_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in data.items():
            if isinstance(v, list):
                overrides += f"{k}='{v}'".replace(" ", "") + " "
            elif v is None:
                overrides += f"{k}=null "
            else:
                overrides += f"{k}={v} "
overrides += f"task_id={args.task_index} "
overrides += f"proj_name={PROJ_NAME} "
if args.try_run:
    overrides += "try_run=True "

# parallel
if args.world_size > 1:
    overrides += f"num_gpus={args.world_size} "
if args.init_method is not None:
    overrides += f"init_method=\"{args.init_method}\" "

if args.cloud:
    pass
else:
    CODE_BASE = os.path.realpath(__file__).split(f"/{PROJ_NAME}")[0]
    print(CODE_BASE)
    param = f"{overrides}"
    print(f"Command param: {param}")
    python = sys.executable

    subprocess.call(
        f"cd {CODE_BASE}/{PROJ_NAME} && " +
        f"{python} testOOD/test_openood.py " + param, shell=True)
