/**
 * 
 */
package neuralNetScript;
import java.util.ArrayList;


/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class RadialBasisFunction implements INodeFunction {

	private static double sigma;
	private double[] associatedCluster;// one of the vectors made via k-means clustering
	//ArrayList<Double> test = new ArrayList<Double>(); //where this is the input of datapoints
	
	@Override
	public double computeOutput(double weightedSum) {
		//need to call RBF here this.rbf(test); //fix here
		return 0;
	}

	// Radial Basis Function
		private double rbf(ArrayList<Double> input) throws Exception{
			double output = -1;
			if(input.size() < 1){
				throw new Exception("The Radial Basis Function needs at least one input");
			}
			for(int i = 0; i < input.size() - 1; i++){
				//double top = (input.get(i)- this.setAssociatedCluster(input.get(i)));  //fix here
				double variance = 2 * Math.pow(sigma, 2); 
				output = Math.exp(-1 * ((Math.pow(top, 2))) / variance);			
			}
			return output;
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
