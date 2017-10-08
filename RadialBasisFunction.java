/**
 * 
 */
package neuralNetScript;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.ListIterator;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class RadialBasisFunction implements INodeFunction {

	public static double sigma;
	private double[] associatedCluster;// one of the vectors made via k-means clustering
	
	@Override
	public double computeOutput(double weightedSum) {
		// TODO Auto-generated method stub
		
		return 0;
	}
	
	// Radial Basis Function
	private static double rbf(ArrayList<Double> input) throws Exception{
		double current = -1;
		double previous = -1;
		if(input.size() < 1){
			throw new Exception("The Radial Basis Function needs at least one input");
		}
		double output = -1;
		ListIterator<Double> iter = input.listIterator();
		while(iter.hasNext()) {
			current = iter.next();
		}
		while(iter.hasPrevious()) {
			previous = iter.previous();
		}
		for(int i = 0; i < input.size() - 1; i++){
			double center = (current-previous); //finds the center of the cluster
			double variance = 2*Math.pow(sigma,2);
			output = Math.exp(-1*((Math.pow(center,2)))/variance);			}
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
