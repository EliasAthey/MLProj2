/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class SigmoidalFunction implements INodeFunction {

	// computes the hyperbolic tangent of the weighted sum for hidden nodes in a multilayer perceptron network
	@Override
	public double computeOutput(double weightedSum) {
		return Math.tanh(weightedSum);
	}
}
