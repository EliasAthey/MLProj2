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
	
	// linearly scales the weightedSum
	@Override
	public double computeOutput(double weightedSum) {
		if (funct.equals(null)) {
			return 5*weightedSum;
		}
		return (funct * weightedSum);
	}
	
	// set the scaling factor
	public void setFunct(double funct) {
		this.funct = funct;
	}

}
