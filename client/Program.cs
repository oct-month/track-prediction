using Thrift;
using Thrift.Collections;

using Thrift.Protocol;
using Thrift.Protocol.Entities;
using Thrift.Protocol.Utilities;
using Thrift.Transport;
using Thrift.Transport.Client;
using Thrift.Transport.Server;
using Thrift.Processor;

public static class Program
{
    public static void Main()
    {

        modeluse model = new modeluse();
        double s = model.forecast_ll("cn", 1, 1, 1, 1, 1, 1, 1);
        Console.WriteLine(s);
    }
}
