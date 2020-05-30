using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace IntelligentSystemsProject.NeuralNetworks.FeedForwardNeuralNetworks
{
    /// <summary>
    /// Represents the artificial input neuron of a feed forward neural network
    /// architecture.
    /// </summary>
    [Serializable]
    class InputNeuronFFNN
    {
        /// <summary>
        /// Gets or sets the value of this input neuron.
        /// </summary>
        public double Data { get; set; }

        /// <summary>
        /// Initializes an instance of an input neuron of a feed forward neural network architecture with a default input value of 0.
        /// </summary>
        public InputNeuronFFNN()
        {
            this.Data = 0;
        }

        /// <summary>
        /// Initializes an instance of an input neuron of a feed forward neural network architecture with a set default input value.
        /// </summary>
        /// <param name="data">Default input value.</param>
        public InputNeuronFFNN(double data)
        {
            this.Data = data;
        }
    }
}
