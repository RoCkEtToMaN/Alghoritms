using Algorithms.Algorithms;
using System;
using System.Linq;

namespace Algorithms
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Quale algortimo vogliamo usare?\n1)Adam\n2)GA");
            int choice = Convert.ToInt32(Console.ReadLine());

            if (Convert.ToInt32(Console.ReadLine()) == 1)
            if (choice == 1)
            {
                // Parametri iniziali (es. per una semplice funzione quadratica)
                double[] parameters = { 3.0, -2.0 };

                // Ottimizzatore Adam
                var optimizer = new Adam();

                // Esempio di loop di ottimizzazione
                for (int i = 0; i < 10000; i++)
                {
                    // Calcolo gradienti (esempio: f(x,y) = x^2 + y^2)
                    double[] gradients = {
                        2 * parameters[0], // Derivata rispetto a x
                        2 * parameters[1]  // Derivata rispetto a y
                    };

                    // Applica l'update di Adam
                    optimizer.Update(parameters, gradients);

                    // Stampa il progresso (opzionale)
                    if (i % 100 == 0)
                    {
                        double loss = parameters.Select(p => p * p).Sum();
                        Console.WriteLine($"Iterazione {i}: Loss = {loss:F6}");
                    }
                }

                Console.WriteLine("Parametri finali: [" +
                    string.Join(", ", parameters.Select(p => p.ToString("F4"))) + "]");
            }
            else if (Convert.ToInt32(Console.ReadLine()) == 2)
            else if (choice == 2)
            {

            }
            
        }
    }
}
