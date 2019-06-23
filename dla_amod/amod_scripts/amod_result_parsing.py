#!/home/scratch.yilinz_t19x/anaconda2/bin/python2.7

_description='''
This tool is used to tuning precision parameters of DLA AMOD.
'''

import os
import inspect
import re
import sys
import argparse
import commands
import math
import logging
from pprint import pprint
from collections import OrderedDict
from google.protobuf import text_format as proto_text


class dict_of_dict(OrderedDict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return OrderedDict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

__version__ = '0.5'


def display_predicted_average(_data_base, fail_ratio):
    for identifier in _data_base.keys():
        logging.info("%s average: %f" % (identifier, _data_base[identifier]['data_average']))

    logging.info('fail ratio: %f' % (fail_ratio))


def calc_prediction_average( _data_base, directory ):
    global _args
    failed_lines = 0;
    total_lines = 0;
    failed_ratio = 0.0
    # TODO: need debug
    source_lines = commands.getoutput("find " + directory + " -name test.stderr | xargs tail -n "\
            + str(_args.line_num) + " | grep -v -E \"==>|loss \"")
    
    if _args.debug_file is not '':
        f = open(_args.debug_file, 'w')
        f.write(source_lines)
        f.close()

    source_lines = source_lines.split("\n")
    assert len(source_lines) >= 3

    line_num = 0
    total_lines = len(source_lines)

    for line in source_lines:
        if re.match(r'^\s*$', line):
            continue

        m = re.search(r".*\] (?P<identifier>[\/\w-]+) = (?P<score>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)", line)
        if m:
            identifier = m.group('identifier')
            score = m.group('score')

            if identifier in _data_base.keys():
                _data_base[identifier]['data_sum'] += float(score)
                _data_base[identifier]['data_count'] += 1
            else:
                ## blob name is not existed, got a new blob name
                _data_base[identifier]['data_sum'] = float(score)
                _data_base[identifier]['data_count'] = 1
        else:
            logging.info("Parse failed in line %d: %s" % (line_num, line))
            failed_lines += 1
        line_num += 1

    item_count = 0
    ## Calculate the average
    for identifier in _data_base.keys():
        _data_base[identifier]['data_average'] =\
                _data_base[identifier]['data_sum']/_data_base[identifier]['data_count']

    display_predicted_average(_data_base, failed_lines/float(total_lines))

def main(argc, argv):
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=_description)
    parser.add_argument('--version', action="version", version=__version__)

    parser.add_argument("--log_dir",
            action="store",
            dest="log_dir",
            required=True,
            help="Directory of result log")
   
    parser.add_argument("--line",
            action="store",
            dest="line_num",
            required=False,
            type=int,
            default=5,
            help="Number of result lines per file: 5 for alexnet, 15 for googlenet")

    parser.add_argument("--debug_file",
            action="store",
            dest="debug_file",
            required=False,
            default='',
            help="All debug information will output to this file")
    parser.add_argument("--log",
            action="store",
            dest="log_file",
            required=False,
            default='',
            help="Debug message will be output to this file")
    
    FORMAT = '%(asctime)-15s %(message)s'
    global _args
    _args = parser.parse_args()
    if _args.log_file:
        logging.basicConfig(format=FORMAT, filename=_args.log_file, filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(format=FORMAT, level=logging.INFO)


    data_base       = dict_of_dict()
    calc_prediction_average(data_base, _args.log_dir)
    return

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
