import logging.handlers
import logging
import sys


class AppLogger:
    def __init__(self, filename: str, level: int) -> None:
        self.log = logging.getLogger()
        fmt = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        handle1 = logging.handlers.RotatingFileHandler(filename, maxBytes=10)
        handle1.setFormatter(fmt)
        # handle2 = logging.StreamHandler(stream=sys.stdout)
        # handle2.setFormatter(fmt)
        self.log.addHandler(handle1)
        # self.log.addHandler(handle2)
        self.log.setLevel(level)
    
    def debug(self, msg) -> None:
        self.log.debug(msg)
    
    def info(self, msg) -> None:
        self.log.info(msg)

    def warn(self, msg) -> None:
        self.log.warning(msg)
    
    def error(self, msg) -> None:
        self.log.error(msg)


logger = AppLogger('app.log', logging.INFO)
