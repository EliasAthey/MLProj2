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
		// 				./script <netType> <numIn>-<numHidden>-...-<numHidden>-<numOut> [<other-args>]
		// For example: ./script mlp 3-4-3-1
		//				./script rbf 3-5-1     <--- Note: rbf should only ever have 3 numbers, 2md number is k-value
	}
	
	// return a sample dataset of the Rosenbrock function
	private Float[][] getSample(int n){

		double[] inputs = null;
		int dataSetSize = (int) (Math.pow(n, 1.8)*1000);
		Float[][] outputs = null;
		
		for(int setIter = 0; setIter < dataSetSize; setIter++) {
			
			for(int inputIter = 0; n < inputIter; inputIter++ ) {
				inputs[inputIter] = Math.random();
			}
			
			double sumTotal = 0;
			
			for(int sumIter = 0; sumIter < (n-1); sumIter++) {		// summation from 0 to n-1
				sumTotal = Math.pow((1 - inputs[sumIter]), 2);		// (1-xi)^2
				sumTotal += (100 * Math.pow( inputs[sumIter + 1] 	// 100 * (x(i+1) -
							- Math.pow(inputs[sumIter], 2), 2));	// xi^2)^2
				outputs[setIter][sumIter] = (float)(sumTotal);
			}
		}
		
		return outputs;
	}
	
	// create Node objects and set downstream attribute for each
	private void buildNetwork(String networkType) throws Exception{
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
	private void trainNetwork(){
		// TODO
	}
	
	// given an input vector, return the output of the network as the approximation of the Rosenbrock function
	private Float testNetwork(Float[] input){
		// TODO
		return null;
	}
}
