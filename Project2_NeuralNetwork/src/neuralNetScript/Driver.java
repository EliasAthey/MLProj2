/**
 * 
 */
package neuralNetScript;

import java.util.ArrayList;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
public class Driver {
	// attributes
	private static String networkType;
	private static int numInNodes;
	private static ArrayList<Integer> numHiddenLayers = new ArrayList<Integer>();// length is # of layers, value @ each index is # of nodes in that layer
	private static int numOutNodes;
	
	// the network itself
	private Layer[] network;
	
	public static void main(String args[]){
		// TODO
		// Input will receive:
		// 				./script <netType> <numIn>-<numHidden>-...-<numHidden>-<numOut> [<other-args>]
		// For example: ./script mlp 3-5-4-1   
		//						--or--
		//				./script rbf 3-5-1     <--- Note: rbf should only ever have 3 numbers, 2md number is k-value
		
		// these will be set according to the input. Example values for now
		Driver.networkType = "mlp";
		Driver.numInNodes = 3;
		Driver.numHiddenLayers.add(5);
		Driver.numHiddenLayers.add(4);
		Driver.numOutNodes = 1;
		
	}
	
	// return a sample dataset of the Rosenbrock function
	private static Float[][] getSample(int size){
		// TODO
		return null;
	}
	
	// the Rosenbrock function
	private static double rosenbrock(ArrayList<Double> input) throws Exception{
		if(input.size() < 2){
			throw new Exception("Rosenbrock function input must have at least two elements.");
		}
		
		double output = 0f;
		for(int i = 0; i < input.size() - 1; i++){
			output += Math.pow(1 - input.get(i), 2) + (100 * Math.pow(input.get(i + 1) - Math.pow(input.get(i), 2), 2));
		}
		
		return output;
	}
	
	// create Node objects and set downstream attribute for each
	private static void buildNetwork(String networkType) throws Exception{
		// TODO
		switch(networkType){
			case "rbf":
				// use k-value to create clusters via k-means clustering; this determines # of hidden nodes
				// create output node
				// create hidden nodes, set downstream to output, set each associatedCluster
				// create input nodes, set the input, set downstream to all nodes in hidden layer
				// set the sigma value for RadialBasisFunction to whatever we want
				RadialBasisFunction.setSigma(2.5f);
				break;
			case "mlp":
				break;
			default: throw new Exception("The specified network type is not defined.");
		}
	}
	
	// input training data into the network, update weights until convergence
	private static void trainNetwork(){
		// TODO
	}
	
	// given an input vector, return the output of the network as the approximation of the Rosenbrock function
	private static double testNetwork(double[] input){
		// TODO
		return 0;
	}
}
