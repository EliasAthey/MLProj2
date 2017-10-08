/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class SigmoidalFunction implements INodeFunction {

	@Override
	public double computeOutput(double weightedSum) {
		
		return 1/(1 + Math.exp(-1 * weightedSum));
	}

}
