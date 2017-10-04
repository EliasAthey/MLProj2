/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class RadialBasisFunction implements INodeFunction {

	private static Float sigma;
	private Float[] associatedCluster;// one of the vectors made via k-means clustering
	
	@Override
	public Float computeOutput(Float weightedSum) {
		// TODO Auto-generated method stub
		return null;
	}

	// sets the sigma value to be used for all Radial Basis Functions
	public static void setSigma(Float sigma){
		RadialBasisFunction.sigma = sigma;
	}
	
	// sets the associated cluster vector
	public void setAssociatedCluster(Float[] cluster){
		this.associatedCluster = cluster;
	}
}
