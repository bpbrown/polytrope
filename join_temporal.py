'''
Join sets of dedalus output data.

Usage:
      join_temporal.py <case>... [--data_type=<data_type> --cleanup]

Options:
      --data_type=<data_type>      Type of data to join; if provided join a single data.
      --cleanup                    Cleanup after join

'''
import glob
import os
import sys
import logging
logger = logging.getLogger(__name__)

import dedalus.public
from dedalus.tools  import post

from docopt import docopt

def join_temporal(base_path,data_types=None, cleanup=False):
    logger.info("joining data in time from Dedalus run {:s}".format(data_dir))
    if data_types is None:
        data_types = ['scalar', 'profiles']
    
    for data_type in data_types:
        logger.info("merging {}".format(data_type))
        try:
            path = base_path+data_type
            files = glob.glob(os.path.join(path,"*.h5"))
            joined_filename = os.path.join(path,"{}_joined.h5".format(data_type))
            post.merge_sets(joined_filename, files, cleanup=cleanup)
        except:
            logger.info("missing {}".format(data_type))
            raise
    
    logger.info("done temporal join operation for {:s}".format(data_dir))

if __name__ == "__main__":

    args = docopt(__doc__)

    data_dir = args['<case>'][0]
    base_path = os.path.abspath(data_dir)+'/'

    cleanup = args['--cleanup']
    data_types = args['--data_type']
    join_temporal(base_path,data_types=data_types,cleanup=cleanup)
