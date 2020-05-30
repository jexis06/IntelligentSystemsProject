using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace IntelligentSystemsProject.NeuralNetworks.FeedForwardNeuralNetworks
{
    /// <summary>
    /// Represents the artificial output neuron of a feed forward neural network
    /// architecture.
    /// </summary>
    [Serializable]
    class OutputNeuronFFNN
    {
        private List<HiddenNeuronFFNN> hiddenNeurons;
        private Random rnd;
        private List<double> weight_o;
        private double bias_o;
        private double delta_o;

        /// <summary>
        /// Gets or sets the output value of this output neuron.
        /// </summary>
        public double Output { get; set; }
        /// <summary>
        /// Gets or sets the desired output value of this output neuron.
        /// </summary>
        public double DesiredOutput { get; set; }
        /// <summary>
        /// Gets or sets the learning rate of this output neuron.
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// Initializes an instance of an output neuron of a feed forward neural network architecture.
        /// </summary>
        /// <param name="learning_rate">Learning rate of the output neuron.</param>
        public OutputNeuronFFNN(double learning_rate)
        {
            this.hiddenNeurons = new List<HiddenNeuronFFNN>();
            this.weight_o = new List<double>();
            this.bias_o = 0;
            this.Output = 0;
            this.delta_o = 0;
            this.LearningRate = learning_rate;
            this.rnd = new Random(DateTime.Now.Millisecond);
        }

        /// <summary>
        /// Adds an instance of a hidden neuron as one of the inputs to this output neuron.
        /// </summary>
        /// <param name="hNeuron">Hidden neuron.</param>
        public void AddHiddenNeuron(HiddenNeuronFFNN hNeuron)
        {
            this.hiddenNeurons.Add(hNeuron);
        }

        /// <summary>
        /// Gets the pos'th instance of a hidden neuron of this output neuron.
        /// </summary>
        /// <param name="pos">The hidden neuron number. Starts at 0.</param>
        /// <returns>The pos'th instance of a hidden neuron of this output neuron.</returns>
        public HiddenNeuronFFNN GetHiddenNeuron(int pos)
        {
            HiddenNeuronFFNN i = null;
            if (pos < this.hiddenNeurons.Count)
            {
                i = this.hiddenNeurons.ElementAt(pos);
            }
            return i;
        }

        /// <summary>
        /// Randomize the weights and bias values of this output neuron. Used for initialization.
        /// </summary>
        public void RandomizeWeightsAndBias()
        {
            double w;
            for (int i = 0; i < this.hiddenNeurons.Count; i++)
            {
                w = (this.rnd.NextDouble() * 2 - 1) / 2;
                this.weight_o.Add(w);
                this.hiddenNeurons.ElementAt(i).OutputWeights.Add(this.weight_o[i]);
            }
            this.bias_o = (this.rnd.NextDouble() * 2 - 1) / 2;
        }

        /// <summary>
        /// Computes the output data of this output neuron.
        /// </summary>
        public void ComputeOutputData()
        {
            this.Output = 0;
            for (int i = 0; i < this.hiddenNeurons.Count; i++)
            {
                this.Output += this.hiddenNeurons.ElementAt(i).Data * this.weight_o.ElementAt(i);
            }
            this.Output += this.bias_o;
            this.Output = HelperFunctions.ComputeSigmoid(this.Output);
        }

        /// <summary>
        /// Computes the delta o of this output neuron.
        /// </summary>
        public void ComputeDeltaO()
        {
            this.delta_o = this.Output * (1 - this.Output) * (this.DesiredOutput - this.Output);
            foreach (HiddenNeuronFFNN hNeuron in this.hiddenNeurons)
            {
                hNeuron.OutputDeltas.Add(this.delta_o);
            }
        }

        /// <summary>
        /// Updates the weights of this hidden neuron.
        /// </summary>
        public void UpdateWeights()
        {
            double[] new_weight_o = new double[this.weight_o.Count];
            for (int i = 0; i < this.weight_o.Count; i++)
            {
                new_weight_o[i] = this.weight_o.ElementAt(i) + (this.LearningRate * this.delta_o * this.hiddenNeurons.ElementAt(i).Data);
                this.hiddenNeurons.ElementAt(i).OutputWeights.Add(new_weight_o[i]);
            }
            this.weight_o = new_weight_o.ToList();
            this.bias_o += this.LearningRate * this.delta_o;
        }

        /// <summary>
        /// Computes the error between the output and the desired output.
        /// </summary>
        /// <returns></returns>
        public double ComputeError()
        {
            return this.DesiredOutput - this.Output;
        }

        /// <summary>
        /// Sets the weight of this output neuron to the pos_i'th hidden neuron.
        /// </summary>
        /// <param name="weight">Weight.</param>
        /// <param name="pos_i">Hidden neuron number. Starts at 0.</param>
        public void SetWeight(double weight, int pos_i)
        {
            double[] weight_o_temp = this.weight_o.ToArray();
            weight_o_temp[pos_i] = weight;
            this.weight_o = weight_o_temp.ToList();
        }

        /// <summary>
        /// Sets the bias value of this output neuron.
        /// </summary>
        /// <param name="bias">Bias value.</param>
        public void SetBias(double bias)
        {
            this.bias_o = bias;
        }

        /// <summary>
        /// Gets the weight of the pos_i'th hidden neuron to this output neuron.
        /// </summary>
        /// <param name="pos_h">Hidden neuron number. Starts at 0.</param>
        /// <returns>Weight.</returns>
        public double GetWeight(int pos_h)
        {
            return this.weight_o.ElementAt(pos_h);
        }

        /// <summary>
        /// Gets the bias value of this output neuron.
        /// </summary>
        /// <returns>Bias value.</returns>
        public double GetBias()
        {
            return this.bias_o;
        }
    }
}
