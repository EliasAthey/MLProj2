/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class RadialBasisFunction implements INodeFunction {

	private static double sigma;
	private double[] associatedCluster;// one of the vectors made via k-means clustering
	
	@Override
	public double computeOutput(double weightedSum) {
		// TODO Auto-generated method stub
		return 0;
	}

	// sets the sigma value to be used for all Radial Basis Functions
	public static void setSigma(double sigma){
		RadialBasisFunction.sigma = sigma;
	}
	
	// sets the associated cluster vector
	public void setAssociatedCluster(double[] cluster){
		this.associatedCluster = cluster;
	}
}
