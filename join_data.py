import os
import sys

from dedalus.tools.logging import logger
from dedalus.tools  import post

data_dir = sys.argv[1]
base_path = os.path.abspath(data_dir)+'/'

logger.info("joining data from Dedalus run {:s}".format(data_dir))
logger.info("merging checkpoint")
post.merge_analysis(base_path+'checkpoint')
logger.info("merging profile_data")
post.merge_analysis(base_path+'profile_data')
logger.info("merging slices")
post.merge_analysis(base_path+'slices')
logger.info("done join operation for {:s}".format(data_dir))
