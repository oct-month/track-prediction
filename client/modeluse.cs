using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Thrift;
using Thrift.Collections;

using Thrift.Protocol;
using Thrift.Protocol.Entities;
using Thrift.Protocol.Utilities;
using Thrift.Transport;
using Thrift.Transport.Client;
using Thrift.Transport.Server;
using Thrift.Processor;

public class modeluse
{
    private TTransport tTransport;
    private intelligence.Client client;

    public modeluse()
    {
        tTransport = new TSocketTransport("127.0.0.1", 3000, null);
        TProtocol protocol = new TBinaryProtocol(tTransport);
        client = new intelligence.Client(protocol);
    }

    ~modeluse()
    {
        if (tTransport.IsOpen)
        {
            tTransport.Close();
        }
    }

    public double forecast_ll(string fn, double longi, double lati, double h, double v, double course, double dlongi, double dlati)
    {
        if (!tTransport.IsOpen)
        {
            tTransport.OpenAsync().GetAwaiter().GetResult();
        }
        return client.forecast_ll(fn, longi, lati, h, v, course, dlongi, dlati).GetAwaiter().GetResult();
    }

    public double forecast_xy(string fn, double x, double y, double h, double v, double course, double dx, double dy)
    {
        if (!tTransport.IsOpen)
        {
            tTransport.OpenAsync().GetAwaiter().GetResult();
        }
        return client.forecast_xy(fn, x, y, h, v, course, dx, dy).GetAwaiter().GetResult();
    }
}
