from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from .api.intelligence import Processor
from .handler import IntelligenceHandler


def get_server(port: int) -> TServer.TSimpleServer:
    handler = IntelligenceHandler()
    processor = Processor(handler)
    transport = TSocket.TServerSocket('0.0.0.0', port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    return server
