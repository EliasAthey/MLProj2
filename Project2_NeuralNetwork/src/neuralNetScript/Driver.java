/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
public class Driver {
	// attributes
	private String networkType;
	private int numInNodes;
	private int[] numHiddenLayers;// length is # of layers, value @ each index is # of nodes in that layer
	private int numOutNodes;
	
	// the network itself
	private Layer[] network;
	
	public static void main(String args[]){
		// TODO
		// Input will receive:
		// 				./script <netType> <numIn>-<numHidden>-...-<numHidden>-<numOut>
		// For example: ./script mlp 3-4-3-1
		//				./script rbf 3-5-1     <--- Note: rbf should only ever have 3 numbers
	}
	
	// return a sample dataset of the Rosenbrock function
	private Float[][] getSample(int size){
		// TODO
		return null;
	}
	
	// create Node objects and set downstream attribute for each
	private void buildNetwork(){
		// TODO
	}
	
	// input training data into the network, update weights until convergence
	private void trainNetwork(){
		// TODO
	}
	
	// given an input vector, return the output of the network as the approximation of the Rosenbrock function
	private Float testNetwork(Float[] input){
		// TODO
		return null;
	}
}
