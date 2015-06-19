import os
import sys
import logging
logger = logging.getLogger(__name__)

import dedalus.public
from dedalus.tools  import post

data_dir = sys.argv[1]
iteration = sys.argv[2]
base_path = os.path.abspath(data_dir)+'/'

logger.info("joining data from Dedalus run {:s}".format(data_dir))

data_types = ['checkpoint', 'profile_data', 'slices', 'scalar']
for data_type in data_types:
    target = "{}/{}_s{}".format(data_type, data_type, iteration)
    logger.info("merging {}".format(target))
    try:
        post.merge_distributed_set(base_path+target)
    except:
        logger.info("target {} not found".format(target))
