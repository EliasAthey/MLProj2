/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class LinearFunction implements INodeFunction {
	
	// the linear scaling factor
	private Double funct;
	
	// linearly scales the weightedSum of the inputs
	@Override
	public double computeOutput(double[][] inputs) {
		// sum weighted inputs
		double sum = 0;
		for (int i = 0; i < inputs[0].length; i++) {
			sum += (inputs[0][i] * inputs[1][i]);
		}
		if (funct.equals(null)) {
			return 5*sum;
		}
		return (funct * sum);
	}
	
	// set the scaling factor
	public void setFunct(double funct) {
		this.funct = funct;
	}

}
