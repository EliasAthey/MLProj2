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
	public Float computeOutput(Float weightedSum) {
		return weightedSum;
	}

}
