/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class PerceptronOutFunction implements INodeFunction {

	// computes the final output for a multilayer perceptron network
	@Override
	public double computeOutput(double weightedSum) {
		return 1/(1+(Math.exp(-weightedSum)));
	}
}
