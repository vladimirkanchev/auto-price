"""Record important messages/warnings/errors into a log folder."""
import logging
import os

import config
# from datetime import datetime

os.makedirs(config.PATH['logpathname'], exist_ok=True)

logging.basicConfig(
    filename=config.PATH['logpathname'] / config.FILE['logfilename'],
    format="[ %(asctime)s ] %(lineno)d %(name)s -%(levelname)s "
    + "- %(message)s",
    level=logging.INFO)


if __name__ == "__main__":
    logging.info("Logging has started")
