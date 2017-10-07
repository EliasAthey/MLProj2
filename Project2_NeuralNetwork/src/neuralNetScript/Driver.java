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
	private static ArrayList<Layer> network;
	
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
	// [m][n] contains m data points, each with n-1 inputs and 1 output
	private static double[][] getSample(int size){
		// TODO
		return null;
	}
	
	// the Rosenbrock function accepting at least 2 inputs
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
	private static void buildNetwork() throws Exception{
		// TODO
		switch(Driver.networkType){
			case "rbf":
				// use k-value to create clusters via k-means clustering; this determines # of hidden nodes
				// create output node
				// create hidden nodes, set downstream to output, set each associatedCluster
				// create input nodes, set the input, set downstream to all nodes in hidden layer
				// set the sigma value for RadialBasisFunction to whatever we want
				RadialBasisFunction.setSigma(2.5);
				break;
				
			case "mlp":
				// create output node
				// create hidden layers in reverse, setting the downstream nodes accordingly
				// create input layer, set downstream nodes to first hidden layer
				
				// initialize Layers and network
				Layer inputLayer = new Layer();
				Layer[] hiddenLayers = new Layer[Driver.numHiddenLayers.size()];
				for(int i = 0; i < hiddenLayers.length; i++){
					hiddenLayers[i] = new Layer();
				}
				Layer outputLayer = new Layer();
				Driver.network = new ArrayList<Layer>();
				Driver.network.add(inputLayer);
				for(Layer layer : hiddenLayers){
					Driver.network.add(layer);
				}
				Driver.network.add(outputLayer);
				
				// create output nodes and store in output layer
				Node[] outputNodes = new Node[Driver.numOutNodes];
				for(int i = 0; i < outputNodes.length; i++){
					// set the node functions for output nodes
					outputNodes[i] = new Node(new PerceptronOutFunction(), new BackpropFinalWeightFunction(), new Node[0]);
				}
				outputLayer.setNodes(outputNodes);
				
				// create hidden layer nodes and store in hidden layer
				Node[] prevHiddenNodes = null;
				for(int i = hiddenLayers.length - 1; i >= 0; i--){
					Node[] hiddenNodes = new Node[Driver.numHiddenLayers.get(i)];
					for(int j = 0; j < hiddenNodes.length; j++){
						// set the node functions for hidden nodes
						if(i == hiddenLayers.length){
							hiddenNodes[j] = new Node(new SigmoidalFunction(), new BackpropHiddenWeightFunction(), outputNodes);
						}
						else{
							hiddenNodes[j] = new Node(new SigmoidalFunction(), new BackpropHiddenWeightFunction(), prevHiddenNodes);
						}
					}
					hiddenLayers[i].setNodes(hiddenNodes);
					prevHiddenNodes = hiddenNodes;
				}
				
				// create input nodes and store in input layer
				Node[] inputNodes = new Node[Driver.numInNodes];
				for(int i = 0; i < inputNodes.length; i++){
					// set the node functions for input nodes
					inputNodes[i] = new Node(new SigmoidalFunction(), new NoWeightFunction(), prevHiddenNodes);
				}
				inputLayer.setNodes(inputNodes);
				break;
				
			default: throw new Exception("The specified network type is not defined.");
		}
	}
	
	// input training data into the network, update weights until convergence
	private static void trainNetwork(){
		double[][] sample = Driver.getSample((int)Math.pow(1.8, Driver.numInNodes) * 1000);
		
		// iterate through each sample point or until convergence
		for(int i = 0; i < sample.length; i++){
			// set inputs for input nodes
			int j = 0;
			while(j < sample[i].length - 1){
				Driver.network.get(0).getNodes()[j].inputs = new double[2][numInNodes];
				Driver.network.get(0).getNodes()[j].inputs[0][j] = sample[i][j];
				j++;
			}
			
			//set the expected output for this sample point
			double expectedOutput = sample[i][j];
			
			// execute the nodes in the network
			for(Layer layer : Driver.network){
				for(Node node : layer.getNodes()){
					node.execute();
				}
			}
			
			// update the weights in the network
			for(int k = Driver.network.size() - 1; k >= 0 ; k--){
				for(Node node : Driver.network.get(k).getNodes()){
					node.updateWeights();
				}
			}
		}
	}
	
	// given an input vector, return the output of the network as the approximation of the Rosenbrock function
	private static double testNetwork(double[] input){
		// TODO
		return 0;
	}
}
