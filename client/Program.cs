using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


public static class Program
{
    private static Random random = new Random();

    public static double randomDouble()
    {
        return random.NextDouble() * 100;
    }

    public static void Main()
    {
        modeluse model = modeluse.getInstance();
        for (int i = 0; i < 20000; i++)
        {
            double s = model.forecast_ll("cn", randomDouble(), randomDouble(), randomDouble(), randomDouble(), randomDouble(), randomDouble(), randomDouble(), randomDouble());
            Thread.Sleep(10);
            Console.WriteLine(s);
        }
        Console.ReadKey();
    }
}

