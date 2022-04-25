import threading
from time import sleep

from service import get_server
from logger import logger


PORT = 3000

class MyThread(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.server = get_server(PORT)
    
    def run(self) -> None:
        logger.info(f"start server on port {PORT}...")
        self.server.serve()
        logger.info("server done")
    
    
if __name__ == '__main__':
    t = MyThread()
    t.start()
    # t.join()
    while True:
        try:
            sleep(1)
        except BaseException as e:
            logger.error(type(e))
            break
    logger.info("done")
