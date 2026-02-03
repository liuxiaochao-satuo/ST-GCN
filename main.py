#!/usr/bin/env python
import argparse
import sys
import os

# 优先使用当前仓库内的 torchlight，而不是系统里可能安装的同名包
repo_dir = os.path.dirname(os.path.abspath(__file__))
local_torchlight_path = os.path.join(repo_dir, 'torchlight')
if local_torchlight_path not in sys.path:
    sys.path.insert(0, local_torchlight_path)

import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    processors['demo_old'] = import_class('processor.demo_old.Demo')
    processors['demo'] = import_class('processor.demo_realtime.DemoRealtime')
    processors['demo_offline'] = import_class('processor.demo_offline.DemoOffline')
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    p.start()
