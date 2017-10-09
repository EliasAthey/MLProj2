/**
 * 
 */
package neuralNetScript;

import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
//import pattern;

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
	private static double convergenceTime;
	private static ArrayList<Double> prevWeights = new ArrayList<Double>();
	private static double[][] sample;
	
	// the network itself
	private static ArrayList<Layer> network;
	
	// package accessible sample expected output
	static double expectedOutput;
	
	public static void main(String args[]){
		Driver.networkType = args[1];
		String[] layers = args[2].split("-");
		
		Driver.numInNodes = Integer.parseInt(layers[0]);
		
		for(int layerMaker = 1; layerMaker < (layers.length - 1); layerMaker++) {
			
			Driver.numHiddenLayers.add(Integer.parseInt(layers[layerMaker]));
		}

		Driver.numOutNodes = Integer.parseInt(layers[(layers.length - 1)]);
		
		//Driver.networkType = "mlp";
		//Driver.numInNodes = 3;
		//Driver.numHiddenLayers.add(5);
		//Driver.numHiddenLayers.add(4);
		//Driver.numOutNodes = 1;
		try{
			Driver.sample = Driver.getSample((int)Math.pow(1.8, Driver.numInNodes) * 1000);
			Driver.buildNetwork();
			Driver.trainNetwork();
		}
		catch(Exception e){
			System.out.println("Error...");
			System.out.println(e.getMessage());
		}
		
		// test the network using 2, 3, 4 as inputs
		double[] in = {2, 3, 4};
		ArrayList<Double> inList = new ArrayList<Double>();
		inList.add(2.0);
		inList.add(3.0);
		inList.add(4.0);
		System.out.println("Network test on {2, 3, 4}:  " + Driver.testNetwork(in)[0]);
		try{
			System.out.println("Actual Rosenbrock value: " + Driver.rosenbrock(inList));
		}
		catch(Exception e){};
	}
	
	// return a sample dataset of the Rosenbrock function
	// [m][n] contains m data points, each with n-1 inputs and 1 output
	private static double[][] getSample(int size){
		double[][] outputs = new double[Driver.numInNodes + 1][size];
		
		// generate *size number of sample data points
		for(int setIter = 0; setIter < size; setIter++) {
			// generate random inputs
			ArrayList<Double> inputs = new ArrayList<Double>();
			for(int inputIter = 0; inputIter < Driver.numInNodes; inputIter++ ) {
				inputs.add(inputIter, Math.random() * 100);
				outputs[inputIter][setIter] = inputs.get(inputIter);
			}
			
			// set the rosenbrock output
			try{
				outputs[Driver.numInNodes][setIter] = Driver.rosenbrock(inputs);
			}
			catch(Exception e){
				System.out.println(e.getMessage());
			}
		}
		
		return outputs;
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
		System.out.println("Building network...\n");
		// TODO
		switch(Driver.networkType){
			case "rbf":
				int k = 3;				
				// use k-value to create clusters via k-means clustering; this determines # of hidden nodes
				// create output node
				// create hidden nodes, set downstream to output, set each associatedCluster
				// create input nodes, set the input, set downstream to all nodes in hidden layer
				// set the sigma value for RadialBasisFunction to whatever we want
				RadialBasisFunction.setSigma(2.5);
				break;
				
			case "mlp":
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
						if(i == hiddenLayers.length - 1){
							hiddenNodes[j] = new Node(new SigmoidalFunction(), new BackpropHiddenWeightFunction(), outputNodes);
							// initialize input arrays with random weights for downstream nodes
							for(int m = 0; m < outputNodes.length; m++){
								outputNodes[m].inputs = new double[2][hiddenNodes.length];
								for(int k = 0; k < outputNodes[m].inputs[1].length; k++){
									outputNodes[m].inputs[1][k] = Math.random();
								}
							}
						}
						else{
							hiddenNodes[j] = new Node(new SigmoidalFunction(), new BackpropHiddenWeightFunction(), prevHiddenNodes);
							// initialize input arrays with random weights for downstream nodes
							for(int m = 0; m < prevHiddenNodes.length; m++){
								prevHiddenNodes[m].inputs = new double[2][hiddenNodes.length];
								for(int k = 0; k < prevHiddenNodes[m].inputs[1].length; k++){
									prevHiddenNodes[m].inputs[1][k] = Math.random();
								}
							}
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
					// initialize input arrays with random weights for downstream nodes
					for(int j = 0; j < prevHiddenNodes.length; j++){
						prevHiddenNodes[j].inputs = new double[2][inputNodes.length];
						for(int k = 0; k < prevHiddenNodes[j].inputs[1].length; k++){
							prevHiddenNodes[j].inputs[1][k] = Math.random();
						}
					}
					// initialize input node weights with 1
					inputNodes[i].inputs = new double[2][1];
					inputNodes[i].inputs[1][0] = 1;
				}
				inputLayer.setNodes(inputNodes);
				break;
				
			default: throw new Exception("The specified network type is not defined.");
		}
	}
	
	// input training data into the network, update weights until convergence
	private static void trainNetwork(){
		System.out.println("Training network...\n");
		
		//start timer
		double startTime = System.currentTimeMillis();
		
		// iterate through each sample point or until convergence
		for(int i = 0; i < Driver.sample[0].length; i++){
			// set inputs for input nodes
			int j = 0;
			while(j < Driver.sample.length - 1){
				Driver.network.get(0).getNodes()[j].inputs[0][0] = Driver.sample[j][i];
				j++;
			}
			
			//set the expected output for this sample point
			Driver.expectedOutput = Driver.sample[i][j];
			
			// execute the nodes in the network
			for(Layer layer : Driver.network){
				for(Node node : layer.getNodes()){
					node.execute();
				}
			}
			
			// save previous weights and update the weights in the network
			for(Layer layer : Driver.network){
				for(Node node : layer.getNodes()){
					for(double weight : node.inputs[1]){
						Driver.prevWeights.add(weight);
					}
				}
			}
			for(int k = Driver.network.size() - 1; k >= 0 ; k--){
				for(Node node : Driver.network.get(k).getNodes()){
					node.updateWeights();
				}
			}
			
			// check convergence
			if(Driver.hasConverged()){
				break;
			}
		}
		
		// save convergence time
		Driver.convergenceTime = System.currentTimeMillis() - startTime;
		System.out.println("Network has been trained in " + Driver.convergenceTime + " milliseconds.\n");
	}
	
	// checks for weight convergence in the network
	private static boolean hasConverged(){
		// get current weights
		ArrayList<Double> allWeights = new ArrayList<Double>();
		for(Layer layer : Driver.network){
			for(Node node : layer.getNodes()){
				for(double weight : node.inputs[1]){
					allWeights.add(weight);
				}
			}
		}
		
		// check convergence of weights to 3 decimal places
		boolean hasConverged = true;
		for(int i = 0; i < allWeights.size(); i++){
			hasConverged &= (int)(allWeights.get(i) * 1000) == (int)(Driver.prevWeights.get(i) * 1000);
		}
		return hasConverged;
	}
	
	// given an input vector, return the output of the network as the approximation of the Rosenbrock function
	private static double[] testNetwork(double[] input){
		// set inputs
		for(int i = 0; i < Driver.network.get(0).getNodes().length; i++){
			Driver.network.get(0).getNodes()[i].inputs[0][0] = input[i];
		}
		
		// execute the nodes in the network
		for(Layer layer : Driver.network){
			for(Node node : layer.getNodes()){
				node.execute();
			}
		}
		
		// get computed output from output nodes
		double[] output = new double[Driver.numOutNodes];
		for(int i = 0; i < output.length; i++){
			output[i] = Driver.network.get(Driver.network.size() - 1).getNodes()[i].getComputedOutput();
		}
		return output;
	}
}
