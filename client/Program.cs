﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Thrift;
using Thrift.Collections;

using Thrift.Protocol;
using Thrift.Transport;


public static class Program
{
    public static void Main()
    {

        modeluse model = new modeluse();
        double s = model.forecast_ll("cn", 1, 1, 1, 1, 1, 1, 1);
        Console.WriteLine(s);
    }
}
