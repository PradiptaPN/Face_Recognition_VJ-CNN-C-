using System;
using System.IO;
using System.Linq;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Serialization;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training;
using ConvNetSharp.Volume;

namespace TrainFERNetwork
{
    class Program
    {
        private readonly CircularBuffer<double> _testAccWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> _trainAccWindow = new CircularBuffer<double>(100);
        private Net<double> _net;
        private Net<double> net;
        private int _stepCount;
        private SgdTrainer<double> _trainer;

        private static void Main()
        {
            var program = new Program();
            program.MnistDemo();
        }

        public void MnistDemo()
        {
            var datasets = new Datasets();
            if (!datasets.Load(100))
            {
                return;
            }

            // Create network
            this._net = new Net<double>();
            //var json2 = File.ReadAllText(@"D:\TA171801038\Expression Recognition\Alpha1\Alpha1\mynetwork.json");
            //net = SerializationExtensions.FromJson<double>(json2);
            this._net.AddLayer(new InputLayer(48, 48, 1));
            this._net.AddLayer(new ConvLayer(3, 3, 8) { Stride = 1, Pad = 2 });
            this._net.AddLayer(new ReluLayer());
            this._net.AddLayer(new PoolLayer(2, 2) { Stride = 2 });
            this._net.AddLayer(new ConvLayer(3, 3, 16) { Stride = 1, Pad = 2 });
            this._net.AddLayer(new ReluLayer());
            this._net.AddLayer(new PoolLayer(3, 3) { Stride = 3 });
            this._net.AddLayer(new FullyConnLayer(10));
            this._net.AddLayer(new SoftmaxLayer(10));

            this._trainer = new SgdTrainer<double>(_net)
            {
                LearningRate = 0.01,
                BatchSize = 20,
                L2Decay = 0.001,
                Momentum = 0.9
            };

            Console.WriteLine("Convolutional neural network learning...[Press any key to stop]");
            do
            {
                var trainSample = datasets.Train.NextBatch(this._trainer.BatchSize);
                Train(trainSample.Item1, trainSample.Item2, trainSample.Item3);

                var testSample = datasets.Test.NextBatch(this._trainer.BatchSize);
                Test(testSample.Item1, testSample.Item3, this._testAccWindow);

                Console.WriteLine("Loss: {0} Train accuracy: {1}% Test accuracy: {2}%", this._trainer.Loss,
                    Math.Round(this._trainAccWindow.Items.Average() * 100.0, 2),
                    Math.Round(this._testAccWindow.Items.Average() * 100.0, 2));

                Console.WriteLine("Example seen: {0} Fwd: {1}ms Bckw: {2}ms", this._stepCount,
                    Math.Round(this._trainer.ForwardTimeMs, 2),
                    Math.Round(this._trainer.BackwardTimeMs, 2));
            } while (!Console.KeyAvailable);


            var json = _net.ToJson();
            System.IO.File.WriteAllText(@"..\..\..\Network\fernetwork.json", json);
            //Console.WriteLine(json);
            // Console.ReadLine();

            Console.WriteLine("-------------------------------------------------------");

            //var json2 = File.ReadAllText(@"D:\TA171801038\Expression Recognition\Alpha1\Alpha1\mynetwork.json");
            //Console.WriteLine(json2);

            /* Net<double> net = SerializationExtensions.FromJson<double>(json2);
             var json3 = net.ToJson();

             if (json == json3)
             {
                 Console.WriteLine("same");
             }else
             {
                 Console.WriteLine("different");
             }*/

            Console.ReadLine();
        }

        private void Test(Volume<double> x, int[] labels, CircularBuffer<double> accuracy, bool forward = true)
        {
            if (forward)
            {
                _net.Forward(x);
            }

            var prediction = _net.GetPrediction();

            for (var i = 0; i < labels.Length; i++)
            {
                accuracy.Add(labels[i] == prediction[i] ? 1.0 : 0.0);
            }
        }

        private void Train(Volume<double> x, Volume<double> y, int[] labels)
        {
            this._trainer.Train(x, y);

            Test(x, labels, this._trainAccWindow, false);

            this._stepCount += labels.Length;
        }
    }
}
