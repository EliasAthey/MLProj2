/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class NoFunction implements INodeFunction {

	// computes nothing, just returns the weighted sum
	@Override
	public double computeOutput(double weightedSum) {
		return weightedSum;
	}

}
