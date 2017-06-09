"""store_metadata.py

Extract metadata about a run from its p0 log file.

Usage:
    store_metadata.py <base_path> [--output=<output> --unjoined]

Options:
    --output=<output>  Output directory; if blank a guess based on likely case name will be made
"""
import csv
import re 
import os
from docopt import docopt

class Metadata():
    def __init__(self, logfilename, cfg_file='metadata.csv'):
        self.logfilename = logfilename
        self.re = {}
        self.dtype = {}
        self.values = {}
        self.cfg_file = cfg_file
        self.read_init()

    def read_init(self):
        with open(self.cfg_file,"r") as csvfile:
            reader = csv.reader(csvfile)
            for r in reader:
                k,re,dtype = r
                self.re[k] = re
                self.dtype[k] = self.dtype_func(dtype)

    def dtype_func(self,dtype_str):
        if dtype_str == 'int':
            dtype = int
        elif dtype_str == 'float':
            dtype = float
        else:
            dtype = str
        return dtype

    def get_metadata(self):
        for k,regexp in self.re.items():
            self.values[k] = self.dtype[k](self.find(regexp))

    def find(self, regexp):
        with open(self.logfilename,"r") as log:
            for l in log:
                m = re.search(regexp,l)
                if m:
                    break
        return m.groups(1)[0]
if __name__ == "__main__":
    args = docopt(__doc__)

    base_path = args['<base_path>']
    logfile = os.path.join(base_path,'logs','dedalus_log_p0.log')
    metadata = Metadata(logfile)
    metadata.get_metadata()

    print(metadata.values)
