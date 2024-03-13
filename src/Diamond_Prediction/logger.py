from loguru import logger
import os
from dotenv import find_dotenv
from datetime import datetime

logger.remove()

filename = datetime.now().strftime("%Y-%m-%d_%H-%M")

logger.add(os.path.join(os.path.dirname(find_dotenv()),'notebooks','logs',filename +'.log'),backtrace=True, 
           format="-"*200 +"\n{time: YYYY-MM-DD HH:mm:ss} | {level} | {module}.{function}.{line} -- {message}", diagnose=True,colorize= False, level="INFO")