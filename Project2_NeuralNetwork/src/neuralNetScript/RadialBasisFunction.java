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
		
		double[] difference = new double[input.size()];
		for(int i = 0; i < input.size() - 1; i++){
			difference[i] = (input.get(i)- this.associatedCluster[i]);		
		}
		double top = 0;
		for(int i = 0; i < difference.length; i++){
			top += Math.pow(difference[i], 2);
		}
		output = Math.exp(-1 * (top / (2 * sigma)));
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
