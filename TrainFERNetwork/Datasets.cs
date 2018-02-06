using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Net;

namespace TrainFERNetwork
{
    internal class Datasets
    {
        private const string mnistFolder = @"..\..\..\Dataset\FERDataset\";
        private const string trainingLabelFile = "train-labels-idx1-ubyte.gz";
        private const string trainingImageFile = "train-images-idx3-ubyte.gz";
        private const string testingLabelFile = "test-labels-idx1-ubyte.gz";
        private const string testingImageFile = "test-images-idx3-ubyte.gz";
        private const string validationLabelFile = "validation-labels-idx1-ubyte.gz";
        private const string validationImageFile = "validation-images-idx3-ubyte.gz";

        public DataSet Train { get; set; }

        public DataSet Validation { get; set; }

        public DataSet Test { get; set; }

        public bool Load(int validationSize = 1000)
        {
            Directory.CreateDirectory(mnistFolder);

            var trainingLabelFilePath = Path.Combine(mnistFolder, trainingLabelFile);
            var trainingImageFilePath = Path.Combine(mnistFolder, trainingImageFile);
            var testingLabelFilePath = Path.Combine(mnistFolder, testingLabelFile);
            var testingImageFilePath = Path.Combine(mnistFolder, testingImageFile);
            var validationLabelFilePath = Path.Combine(mnistFolder, validationLabelFile);
            var validationImageFilePath = Path.Combine(mnistFolder, validationImageFile);


            // Load data
            Console.WriteLine("Loading the datasets...");
            var train_images = MnistReader.Load(trainingLabelFilePath, trainingImageFilePath);
            var testing_images = MnistReader.Load(testingLabelFilePath, testingImageFilePath);
            var validation_images = MnistReader.Load(validationLabelFilePath, validationImageFilePath);

            if (train_images.Count == 0 || validation_images.Count == 0 || testing_images.Count == 0)
            {
                Console.WriteLine("Missing Mnist training/testing files.");
                Console.ReadKey();
                return false;
            }

            this.Train = new DataSet(train_images);
            this.Validation = new DataSet(validation_images);
            this.Test = new DataSet(testing_images);

            return true;
        }
    }
}
