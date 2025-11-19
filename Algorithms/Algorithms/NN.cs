using System;
using System.Collections.Generic;
using System.Text;

namespace Algorithms.Algorithms
{
    internal class NN
    {
        private readonly int[] _layers;
        private readonly float[][] _neurons;
        private readonly float[][][] _weights;
        private readonly Random _random;


        public NN(int[] layers)
        {
            _layers = layers;
            _random = new Random();

            // Inizializzazione neuroni
            _neurons = new float[layers.Length][];
            for (int i = 0; i < layers.Length; i++)
            {
                _neurons[i] = new float[layers[i]];
            }

            // Inizializzazione pesi
            _weights = new float[layers.Length - 1][][];
            for (int i = 0; i < layers.Length - 1; i++)
            {
                _weights[i] = new float[layers[i + 1]][];
                for (int j = 0; j < layers[i + 1]; j++)
                {
                    _weights[i][j] = new float[layers[i]];
                    for (int k = 0; k < layers[i]; k++)
                    {
                        _weights[i][j][k] = (float)(_random.NextDouble() * 2 - 1); // Pesi tra -1 e 1
                    }
                }
            }
        }
        private static float Sigmoid(float x) => 1.0f / (1.0f + (float)Math.Exp(-x));
        private static float SigmoidDerivative(float x) => x * (1 - x);
        public float[] FeedForward(float[] inputs)
        {
            Array.Copy(inputs, _neurons[0], inputs.Length);

            for (int i = 1; i < _layers.Length; i++)
            {
                for (int j = 0; j < _neurons[i].Length; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < _neurons[i - 1].Length; k++)
                    {
                        sum += _weights[i - 1][j][k] * _neurons[i - 1][k];
                    }
                    _neurons[i][j] = Sigmoid(sum);
                }
            }
            return _neurons[^1];
        }

        public void Train(float[] inputs, float[] targets, float learningRate = 0.1f)
        {
            FeedForward(inputs);

            // Backpropagation
            float[][] errors = new float[_layers.Length][];
            for (int i = 0; i < _layers.Length; i++)
            {
                errors[i] = new float[_layers[i]];
            }

            // Calcolo errore output layer
            for (int i = 0; i < _neurons[^1].Length; i++)
            {
                errors[^1][i] = (targets[i] - _neurons[^1][i]) * SigmoidDerivative(_neurons[^1][i]);
            }

            // Propagazione errori all'indietro
            for (int i = _layers.Length - 2; i >= 0; i--)
            {
                for (int j = 0; j < _neurons[i].Length; j++)
                {
                    float error = 0;
                    for (int k = 0; k < _neurons[i + 1].Length; k++)
                    {
                        error += errors[i + 1][k] * _weights[i][k][j];
                    }
                    errors[i][j] = error * SigmoidDerivative(_neurons[i][j]);
                }
            }

            // Aggiornamento pesi
            for (int i = 0; i < _weights.Length; i++)
            {
                for (int j = 0; j < _weights[i].Length; j++)
                {
                    for (int k = 0; k < _weights[i][j].Length; k++)
                    {
                        _weights[i][j][k] += learningRate * errors[i + 1][j] * _neurons[i][k];
                    }
                }
            }
        }
    }
}

