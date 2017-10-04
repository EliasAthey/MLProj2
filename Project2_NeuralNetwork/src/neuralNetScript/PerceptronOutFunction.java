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
	public Float computeOutput(Float[] args) {
		Float sum = null;
		for (int i = 0; i < args.length; i++) {
			sum += args[i];
		}
		float out = (float)(1/(1+(Math.exp(-sum))));
		return out;
		
	}

}
