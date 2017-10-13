/**
 * 
 */
package neuralNetScript;


/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class RadialBasisFunction implements INodeFunction {

	// the parameter sigma - the variance
	private static double sigma;
	
	// one of the vectors made via k-means clustering
	private Double[] associatedCluster;
	
	// computes the output for Radial Basis Function nodes
	@Override
	public double computeOutput(double[][] inputs) {
		if(inputs[0].length < 1){
			System.out.println("Error: The Radial Basis Function needs at least one input");
		}
		
		double[] difference = new double[inputs[0].length];
		for(int i = 0; i < inputs[0].length; i++){
			difference[i] = (inputs[0][i]- this.associatedCluster[i]);		
		}
		double top = 0;
		for(int i = 0; i < difference.length; i++){
			top += Math.pow(difference[i], 2);
		}
		return Math.exp(-1 * (top / (2 * sigma)));
	}
	
	// sets the sigma value to be used for all Radial Basis Functions
	public static void setSigma(double sigma){
		RadialBasisFunction.sigma = sigma;
	}
	
	// sets the associated cluster vector  
	public void setAssociatedCluster(Double[] cluster){
		this.associatedCluster = cluster;
	}
}
