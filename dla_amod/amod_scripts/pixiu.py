#!/home/scratch.yilinz_t19x/anaconda2/bin/python2.7

_description='''
This tool is used to run the failed tests after NVBatch
'''

import os
import inspect
import re
import sys
import argparse
import commands
import math
import logging
import copy
import subprocess
from subprocess import PIPE
from pprint import pprint
from collections import OrderedDict


cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder += "/../build/python/caffe/"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

proto_dir = r"/home/utils/python-protobuf-2.6.1/lib/python2.7/site-packages/"
if proto_dir not in sys.path:
    sys.path.insert(0, proto_dir)



class dict_of_dict(OrderedDict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return OrderedDict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

__version__ = '0.5'


#################### Global Variables ######################
TOTAL_PROCESS_ON_FLYING=3

def main(argc, argv):
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=_description)
    parser.add_argument('--version', action="version", version=__version__)

    parser.add_argument("--dir",
            action="store",
            dest="test_dir",
            required=True,
            help="The nvbatch test directory")

    parser.add_argument("--amod_dir",
            action="store",
            dest="amod_dir",
            required=True,
            help="The amod directory")

    parser.add_argument("--process",
            action="store",
            dest="num_proc",
            type=int,
            required=True,
            help="Number of process started by nvbatch")

    parser.add_argument("--log",
            action="store",
            dest="log_file",
            required=False,
            default='',
            help="Debug message will be output to this file")

    global _args

    FORMAT = '%(asctime)-15s %(message)s'
    _args = parser.parse_args()
    if _args.log_file:
        logging.basicConfig(format=FORMAT, filename=_args.log_file, filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(format=FORMAT, level=logging.INFO)

    logging.info('start working on batch:%s', _args.test_dir)
    failed_jobs = open('failed_jobs.txt', 'w')
    refresh_jobs = open('refresh.sh', 'w')
    num_failed_jobs = 0
    for i in range(_args.num_proc):
        batch_dir = os.path.join(_args.test_dir, 'batch_'+str(i) )
        testout_file = os.path.join(batch_dir, 'test.stdout')
        # assuem there'll be keyword "PASS" if one job is finished successfully
        p1 = subprocess.Popen(['grep', 'PASS', testout_file], stdout=PIPE)
        p2 = subprocess.Popen(['wc', '-l'], stdin=p1.stdout, stdout=PIPE)
        p1.stdout.close()
        p2.wait()
        p1.wait()
        result = int(p2.communicate()[0])
        if result == 0:
            exe = os.path.join(batch_dir, 'test.sh')
            log = os.path.join(batch_dir, 'test.stderr')
            cmd = exe + ' >& ' + log + '\n'
            failed_jobs.write(cmd)

            cmd = 'cp ./rerun/batch_'  + str(num_failed_jobs) + '/test.stdout ' + os.path.join(batch_dir, 'test.stdout') + '\n'
            refresh_jobs.write(cmd)
            num_failed_jobs = num_failed_jobs+1
            logging.info('#%d job failed, logged to failed_jobs.txt', i )
    failed_jobs.close()
    refresh_jobs.close()

    # Rerun failed tests
    cmd = os.path.join(_args.amod_dir, 'amod_scripts/submit_customized_jobs.pl') + ' -job_file ./failed_jobs.txt -out_dir ./rerun'
    os.system(cmd)
    logging.info('resubmitting failed jobs done!')

    # update the test.stdout
    os.system('sh ./refresh.sh')           
    logging.info('refresh test.stdout done!')

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
