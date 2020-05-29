using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace IntelligentSystemsProject.StorageClasses
{
    /// <summary>
    /// Represents as storage of input-desired_output pairs for the training of
    /// any type of Feed Forward Neural Network architecture.
    /// </summary>
    [Serializable]
    class FeedForwardNNTrainingRules
    {
        List<double[]> input;
        List<double[]> expected_output;

        /// <summary>
        /// Gets the number of input-desired_output pairs in the set of FeedForwardNNTrainingRules.
        /// </summary>
        public int Count { get; private set; }

        /// <summary>
        /// Initializes a new instance of FeedForwardNNTrainingRules which will serve as
        /// storage of input-desired_output pairs for the training of any type of 
        /// Feed Forward Neural Network architecture.
        /// </summary>
        public FeedForwardNNTrainingRules()
        {
            this.input = new List<double[]>();
            this.expected_output = new List<double[]>();
            this.Count = 0;
        }

        /// <summary>
        /// Add input-desired_output pair into the set as rules.
        /// </summary>
        /// <param name="input_data">An array of [double] input data.</param>
        /// <param name="expected_output_data">An array of [double] output data.</param>
        public void addRules(double[] input_data, double[] expected_output_data)
        {
            this.input.Add(input_data);
            this.expected_output.Add(expected_output_data);
        }

        /// <summary>
        /// Clears all the input-desired_output pairs stored in the FeedForwardNNTrainingRules instance.
        /// </summary>
        public void Clear()
        {
            this.input.Clear();
            this.expected_output.Clear();
            this.Count = 0;
        }

        /// <summary>
        /// Gets the input array values of a specific rule number.
        /// </summary>
        /// <param name="rule_number">Rule number. The first rule is numbered 0.</param>
        /// <returns>The input array values of a specified rule number.</returns>
        public double[] getInput(int rule_number)
        {
            return this.input[rule_number];
        }

        /// <summary>
        /// Gets the desired output array values of a specific rule number.
        /// </summary>
        /// <param name="rule_number">Rule number. The first rule is numbered 0.</param>
        /// <returns>The desired output array values of a specified rule number.</returns>
        public double[] getExpectedOutput(int rule_number)
        {
            return this.expected_output[rule_number];
        }
    }
}
