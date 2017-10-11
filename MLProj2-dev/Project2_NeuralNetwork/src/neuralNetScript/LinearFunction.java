/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class LinearFunction implements INodeFunction {
	
	private Double funct;
	
	@Override
	public double computeOutput(double weightedSum) {
		if (funct.equals(null)) {
			return 5*weightedSum;
		}
		return (funct * weightedSum);
	}
	
	public void setFunct(double funct) {
		this.funct = funct;
	}

}
