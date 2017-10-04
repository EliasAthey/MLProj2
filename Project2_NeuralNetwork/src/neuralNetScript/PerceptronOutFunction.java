/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class PerceptronOutFunction implements INodeFunction {

	@Override
	public Float computeOutput(Float weightedSum, Float[] args) {
		float out = (float)(1/(1+(Math.exp(-weightedSum))));
		return out;
	}
}
