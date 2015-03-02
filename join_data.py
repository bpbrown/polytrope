import os
import sys
import logging
logger = logging.getLogger(__name__)

import dedalus.public
from dedalus.tools  import post

data_dir = sys.argv[1]
base_path = os.path.abspath(data_dir)+'/'

logger.info("joining data from Dedalus run {:s}".format(data_dir))
logger.info("merging checkpoint")
post.merge_analysis(base_path+'checkpoint')
logger.info("merging scalar")
post.merge_analysis(base_path+'scalar')
logger.info("merging profile")
post.merge_analysis(base_path+'profile')
logger.info("merging slices")
post.merge_analysis(base_path+'slices')
logger.info("done join operation for {:s}".format(data_dir))
