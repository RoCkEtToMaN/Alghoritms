
using System;

namespace Algorithms.Algorithms
{
    public class Adam
    {
        public double LearningRate { get; set; } = 0.001;
        public double Beta1 { get; set; } = 0.9;
        public double Beta2 { get; set; } = 0.999;
        public double Epsilon { get; set; } = 1e-8;

        private double[] m; // Primo momento (media)
        private double[] v; // Secondo momento (varianza non corretta)
        private int t = 0; // Contatore iterazioni

        public void Update(double[] parameters, double[] gradients)
        {
            t++; // Incrementa contatore ad ogni update

            // Inizializza momenti al primo update
            if (m == null || v == null)
            {
                m = new double[parameters.Length];
                v = new double[parameters.Length];
            }

            for (int i = 0; i < parameters.Length; i++)
            {
                // Aggiorna i momenti
                m[i] = Beta1 * m[i] + (1 - Beta1) * gradients[i];
                v[i] = Beta2 * v[i] + (1 - Beta2) * Math.Pow(gradients[i], 2);

                // Calcola le stime corrette
                double mHat = m[i] / (1 - Math.Pow(Beta1, t));
                double vHat = v[i] / (1 - Math.Pow(Beta2, t));

                // Aggiorna i parametri
                parameters[i] -= LearningRate * mHat / (Math.Sqrt(vHat) + Epsilon);
            }
        }
    }
}
