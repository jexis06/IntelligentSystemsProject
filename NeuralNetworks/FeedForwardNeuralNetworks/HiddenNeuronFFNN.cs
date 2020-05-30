using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace IntelligentSystemsProject.NeuralNetworks.FeedForwardNeuralNetworks
{
    /// <summary>
    /// Represents the artificial hidden neuron of a feed forward neural network
    /// architecture.
    /// </summary>
    [Serializable]
    class HiddenNeuronFFNN:InputNeuronFFNN
    {
        private List<InputNeuronFFNN> inputNeurons;
        private Random rnd;
        private List<double> weight_h;
        private double delta_h;

        /// <summary>
        /// Gets or sets the bias of this hidden neuron.
        /// </summary>
        public double Bias { get; set; }
        /// <summary>
        /// Gets or sets the learning rate of this hidden neuron.
        /// </summary>
        public double LearningRate { get; set; }
        /// <summary>
        /// Gets or sets the array of output deltas of this hidden neuron.
        /// </summary>
        public List<double> OutputDeltas { get; set; }
        /// <summary>
        /// Gets or sets the array of output weights of this hidden neuron.
        /// </summary>
        public List<double> OutputWeights { get; set; }

        /// <summary>
        /// Initializes an instance of a hidden neuron of a feed forward neural network architecture.
        /// </summary>
        /// <param name="learning_rate">Learning rate of the hidden neuron.</param>
        public HiddenNeuronFFNN(double learning_rate) : base()
        {
            this.inputNeurons = new List<InputNeuronFFNN>();
            this.weight_h = new List<double>();
            this.OutputDeltas = new List<double>();
            this.OutputWeights = new List<double>();
            this.rnd = new Random(DateTime.Now.Millisecond);
            this.LearningRate = learning_rate;
        }

        /// <summary>
        /// Adds an instance of an input neuron as one of the inputs to this hidden neuron.
        /// </summary>
        /// <param name="iNeuron">Input neuron instance.</param>
        public void AddInputNeuron(InputNeuronFFNN iNeuron)
        {
            this.inputNeurons.Add(iNeuron);
        }

        /// <summary>
        /// Gets the pos'th instance of an input neuron of this hidden neuron.
        /// </summary>
        /// <param name="pos">The input neuron number. Starts at 0.</param>
        /// <returns>The pos'th instance of an input neuron of this hidden neuron.</returns>
        public InputNeuronFFNN GetInputNeuron(int pos)
        {
            InputNeuronFFNN i = null;
            if (pos < this.inputNeurons.Count)
            {
                i = this.inputNeurons.ElementAt(pos);
            }
            return i;
        }

        /// <summary>
        /// Randomize the weights and bias values of this hidden neuron. Used for initialization.
        /// </summary>
        public void RandomizeWeightsAndBias()
        {
            double w;
            for (int i = 0; i < this.inputNeurons.Count; i++)
            {
                w = (this.rnd.NextDouble() * 2 - 1) / 2;
                this.weight_h.Add(w);
            }
            this.Bias = (this.rnd.NextDouble() * 2 - 1) / 2;
        }

        /// <summary>
        /// Computes the output data of this hidden neuron.
        /// </summary>
        public void ComputeOutputData()
        {
            base.Data = 0;
            for (int i = 0; i < this.inputNeurons.Count; i++)
            {
                base.Data += this.inputNeurons.ElementAt(i).Data * this.weight_h.ElementAt(i);
            }
            base.Data += this.Bias;
            base.Data = HelperFunctions.ComputeSigmoid(base.Data);
        }

        /// <summary>
        /// Computes the delta h of this hidden neuron.
        /// </summary>
        public void ComputeDeltaH()
        {
            double d = 0;
            for (int i = 0; i < this.OutputDeltas.Count; i++)
            {
                d += this.OutputDeltas.ElementAt(i) * this.OutputWeights.ElementAt(i);
            }
            this.delta_h = base.Data * (1 - base.Data) * d;
        }

        /// <summary>
        /// Updates the weights of this hidden neuron.
        /// </summary>
        public void UpdateWeights()
        {
            double[] new_weight_h = new double[this.weight_h.Count];
            for (int i = 0; i < this.weight_h.Count; i++)
            {
                new_weight_h[i] = this.weight_h.ElementAt(i) + (this.LearningRate * this.delta_h * this.inputNeurons.ElementAt(i).Data);
            }
            this.weight_h = new_weight_h.ToList();
            this.Bias += this.LearningRate * this.delta_h;
        }

        /// <summary>
        /// Sets the weight of this hidden neuron to the pos_i'th input neuron.
        /// </summary>
        /// <param name="weight">Weight.</param>
        /// <param name="pos_i">Input neuron number. Starts at 0.</param>
        public void SetWeight(double weight, int pos_i)
        {
            double[] weight_h_temp = this.weight_h.ToArray();
            weight_h_temp[pos_i] = weight;
            this.weight_h = weight_h_temp.ToList();
        }

        /// <summary>
        /// Sets the bias value of this hidden neuron.
        /// </summary>
        /// <param name="bias">Bias value.</param>
        public void SetBias(double bias)
        {
            this.Bias = bias;
        }

        /// <summary>
        /// Gets the weight of the pos_i'th input neuron to this hidden neuron.
        /// </summary>
        /// <param name="pos_i">Input neuron number. Starts at 0.</param>
        /// <returns>Weight.</returns>
        public double GetWeight(int pos_i)
        {
            return this.weight_h.ElementAt(pos_i);
        }

        /// <summary>
        /// Gets the bias value of this hidden neuron.
        /// </summary>
        /// <returns>Bias value.</returns>
        public double GetBias()
        {
            return this.Bias;
        }
    }
}
