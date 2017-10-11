/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class NoFunction implements INodeFunction {

	@Override
	public double computeOutput(double weightedSum) {
		return weightedSum;
	}

}
