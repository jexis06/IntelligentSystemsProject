using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace IntelligentSystemsProject.NeuralNetworks
{
    class HelperFunctions
    {
        public static double ComputeSigmoid(double v)
        {
            return (1 / (1 + Math.Pow(Math.E, (-v))));
        }
    }
}
