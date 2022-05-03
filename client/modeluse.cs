using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

using Thrift;
using Thrift.Collections;

using Thrift.Protocol;
using Thrift.Transport;


public class modeluse
{
    private static modeluse instance = new modeluse();

    private static TTransport tTransport = new TSocket("127.0.0.1", 3000);
    private static intelligence.Client client = new intelligence.Client(new TBinaryProtocol(tTransport));

    private static bool runflag_ll = false;
    private static bool runflag_xy = false;
    private static double result_ll = 0;
    private static double result_xy = 0;

    // 单例模式
    public static modeluse getInstance()
    {
        if (!tTransport.IsOpen)
        {
            tTransport.Open();
        }
        return instance;
    }

    private modeluse()
    {
    }

    ~modeluse()
    {
        if (tTransport != null)
        {
            tTransport.Close();
        }
    }

    class StartForcecast
    {
        private intelligence.Client client;

        private string fn;
        private double t;
        private double x;
        private double y;
        private double h;
        private double v;
        private double course;
        private double dx;
        private double dy;

        public StartForcecast(intelligence.Client client, string fn, double t, double x, double y, double h, double v, double course, double dx, double dy)
        {
            this.client = client;
            this.fn = fn;
            this.t = t;
            this.x = x;
            this.y = y;
            this.h = h;
            this.v = v;
            this.course = course;
            this.dx = dx;
            this.dy = dy;
        }

        public void forecast_ll()
        {
            if (!runflag_ll)
            {
                runflag_ll = true;
                try
                {
                    result_ll = client.forecast_ll(fn, t, x, y, h, v, course, dx, dy);
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.ToString());
                }
                runflag_ll = false;
            }
        }

        public void forcecast_xy()
        {
            if (!runflag_xy)
            {
                runflag_xy = true;
                try
                {
                    result_xy = client.forecast_xy(fn, t, x, y, h, v, course, dx, dy);
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.ToString());
                }
                runflag_xy = false;
            }
        }
    }

    public double forecast_ll(string fn, double t, double longi, double lati, double h, double v, double course, double dlongi, double dlati)
    {
        if (tTransport.IsOpen)
        {
            StartForcecast start = new StartForcecast(client, fn, t, longi, lati, h, v, course, dlongi, dlati);
            ThreadStart childRef = new ThreadStart(start.forecast_ll);
            Thread childThread = new Thread(childRef);
            childThread.IsBackground = true;
            childThread.Start();
        }
        return result_ll;
    }

    public double forecast_xy(string fn, double t, double x, double y, double h, double v, double course, double dx, double dy)
    {
        if (tTransport.IsOpen)
        {
            StartForcecast start = new StartForcecast(client, fn, t, x, y, h, v, course, dx, dy);
            ThreadStart childRef = new ThreadStart(start.forcecast_xy);
            Thread childThread = new Thread(childRef);
            childThread.IsBackground = true;
            childThread.Start();
        }
        return result_xy;
    }
}
