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
	private static Double[][] sample;
	private static int k;
	
	// the network itself
	private static ArrayList<Layer> network;
	
	// package accessible sample expected output
	static double expectedOutput;
	
	public static void main(String args[]){
		Driver.networkType = args[0];
		String[] layers = args[1].split("-");
		
		Driver.numInNodes = Integer.parseInt(layers[0]);
		
		for(int layerMaker = 1; layerMaker < (layers.length - 1); layerMaker++) {
			
			Driver.numHiddenLayers.add(Integer.parseInt(layers[layerMaker]));
		}

		Driver.numOutNodes = Integer.parseInt(layers[(layers.length - 1)]);
		
		try{
			Driver.sample = Driver.getSample((int)Math.pow(1.8, Driver.numInNodes) * 1000);
			
			Driver.buildNetwork();
			Driver.trainNetwork();
		}
		catch(NullPointerException e){
			System.out.println("Error...");
			System.out.println(e.getMessage());
		}
		catch(Exception e){
			System.out.println("Error...");
			System.out.println(e.getMessage());
		}
		
		// test the network using 1, 2, 3 as inputs
		double[] in = {1, 2, 3};
		ArrayList<Double> inList = new ArrayList<Double>();
		inList.add(1.0);
		inList.add(2.0);
		inList.add(3.0);
		System.out.println("Network test on {1, 2, 3}:  " + Driver.testNetwork(in)[0]);
		try{
			System.out.println("Actual Rosenbrock value: " + Driver.rosenbrock(inList));
		}
		catch(Exception e){};
	}
	
	// return a sample dataset of the Rosenbrock function
	// [m][n] contains m data points, each with n-1 inputs and 1 output
	private static Double[][] getSample(int size){
		Double[][] outputs = new Double[Driver.numInNodes + 1][size];
		
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
		// print status message and model visualization
		System.out.println("Building network...");
		System.out.print(Driver.numInNodes + "(in) -> ");
		for(int i = 0; i < Driver.numHiddenLayers.size(); i++){
			System.out.print(Driver.numHiddenLayers.get(i) + " -> ");
		}
		System.out.print(Driver.numOutNodes + "(out)\n\n");
		
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
			switch(Driver.networkType){
				case "rbf":
					outputNodes[i] = new Node(new NoFunction(), new BackpropFinalWeightFunction(), new Node[0]);
					break;
				case "mlp": 
					outputNodes[i] = new Node(new PerceptronOutFunction(), new BackpropFinalWeightFunction(), new Node[0]);
					break;
				default: throw new Exception("The specified network type is not defined.");
			}
			outputNodes[i].setLayerIndex(i);
		}
		outputLayer.setNodes(outputNodes);
		
		// create hidden layer nodes and store in hidden layer
		Node[] prevHiddenNodes = null;
		for(int i = hiddenLayers.length - 1; i >= 0; i--){
			Node[] hiddenNodes = new Node[Driver.numHiddenLayers.get(i)];
			for(int j = 0; j < hiddenNodes.length; j++){
				// set the node functions for hidden nodes
				if(i == hiddenLayers.length - 1){
					switch(Driver.networkType){
						case "rbf": 
							hiddenNodes[j] = new Node(new RadialBasisFunction(), new NoWeightFunction(), outputNodes);
							Driver.k = Driver.numHiddenLayers.get(0);

							// TODO need to store the clusters. This should return an ArrayList<double[]>
							ArrayList<Double[]> clusters = kmeans();
							hiddenNodes[j].setAssociatedCluster(clusters.get(j));
							// set the sigma value for RadialBasisFunction to whatever we want
							RadialBasisFunction.setSigma(2.5);
							break;
						case "mlp":
							hiddenNodes[j] = new Node(new SigmoidalFunction(), new BackpropHiddenWeightFunction(), outputNodes);
							break;
					}
					
					// initialize input arrays with random weights for downstream nodes
					for(int m = 0; m < outputNodes.length; m++){
						outputNodes[m].inputs = new double[2][hiddenNodes.length];
						for(int k = 0; k < outputNodes[m].inputs[1].length; k++){
							outputNodes[m].inputs[1][k] = Math.random();
						}
					}
				}
				else{
					if(Driver.networkType.equals("rbf")){
						throw new Exception("An rbf network cannot have more than one hidden layer.");
					}
					hiddenNodes[j] = new Node(new SigmoidalFunction(), new BackpropHiddenWeightFunction(), prevHiddenNodes);
					// initialize input arrays with random weights for downstream nodes
					for(int m = 0; m < prevHiddenNodes.length; m++){
						prevHiddenNodes[m].inputs = new double[2][hiddenNodes.length];
						for(int k = 0; k < prevHiddenNodes[m].inputs[1].length; k++){
							prevHiddenNodes[m].inputs[1][k] = Math.random();
						}
					}
				}
				hiddenNodes[j].setLayerIndex(j);
			}
			hiddenLayers[i].setNodes(hiddenNodes);
			prevHiddenNodes = hiddenNodes;
		}
		
		// create input nodes and store in input layer
		Node[] inputNodes = new Node[Driver.numInNodes];
		for(int i = 0; i < inputNodes.length; i++){
			// set the node functions for input nodes
			switch(Driver.networkType){
				case "rbf": 
					inputNodes[i] = new Node(new NoFunction(), new NoWeightFunction(), prevHiddenNodes);
					break;
				case "mlp":
					inputNodes[i] = new Node(new SigmoidalFunction(), new NoWeightFunction(), prevHiddenNodes);
					break;
			}
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
			inputNodes[i].setLayerIndex(i);
		}
		inputLayer.setNodes(inputNodes);
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
			Driver.expectedOutput = Driver.sample[j][i];
			
			// execute the nodes in the network
			for(Layer layer : Driver.network){
				for(Node node : layer.getNodes()){
					node.execute();
				}
			}
			
			// save previous weights to test convergence, then update the weights in the network
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

	// given a k and the training set return k centroids that
	// define the centers of the clusters
	private static ArrayList<Double[]> kmeans(){
		ArrayList<Double[]> centroids = new ArrayList<Double[]>(Driver.k);
		int[] labels = new int[Driver.sample[0].length];
		
		// pick initial random data points to be centroids
		for(int randClusterIter = 0; randClusterIter < Driver.k; randClusterIter++) {
			Double[] randCentroid = new Double[Driver.numInNodes];
			int randIndex = (int) (Math.random() * Driver.sample[0].length);
			
			for(int randSampler = 0; randSampler < Driver.numInNodes; randSampler++) {
				randCentroid[randSampler] = sample[randSampler] [randIndex]; 
			}
			centroids.add(randCentroid);
		}
		
		int iterations = 0;
		ArrayList<Double[]> oldCentroids = null;
		
		do{
			
			// save old for convergence test
			oldCentroids = centroids;
			iterations ++;
			labels = getLabels(centroids);
			centroids = getNewCentroids(centroids, labels);
		}while (!stopKmeans(oldCentroids, centroids));
		
		return centroids;
	}
	
	// assigns a label for every datapoint in the sample set
	// uses distance function to find closest centroid
	// label is index of centroid in centroids[]
	private static int[] getLabels(ArrayList<Double[]> centroids) {
		
		int[] labels = new int[Driver.sample[0].length];
		Double[] distances = new Double[Driver.numInNodes];
		Double sum = null;
		
		for(int sampleIter = 0; sampleIter < Driver.sample[0].length; sampleIter++) {//loop thru samples
			
			for (int centroidIter = 0; centroidIter < Driver.k; centroidIter++) {//loop thru centroids
				sum = (double) 0;
				
				for (int dimensionIter = 0; dimensionIter < Driver.numInNodes; dimensionIter++){//loop thru dimensions of centroids
					//sum for Euclidean distance function
					sum += Math.pow(Driver.sample[dimensionIter][sampleIter] - centroids.get(centroidIter)[dimensionIter], 2);
				}
				//calc distance to each centroid from each sample
				distances[centroidIter] = Math.sqrt(sum);
			}
			//store index of min distance in labels
			labels[sampleIter] = findMin(distances);
		}
		return labels;
	}
	
	// calculate geometric mean of all sample points with a common label
	// make this point a new centroid
	// dimensionSums[l][d] holds the label, [l] with the sum of all 
	// dimensions of all samples with that label, [d]
	private static ArrayList<Double[]> getNewCentroids(ArrayList<Double[]> centroids, int[] labels) {
		
		
		ArrayList<Double[]> newCentroids = new ArrayList<Double[]>(Driver.k);
		int[] labelDivisors = new int[Driver.k];
		Double[][] dimensionSums = new Double[Driver.k][Driver.numInNodes];
		// initialize dimensionSums to 0
		for (int i = 0; i < dimensionSums.length; i++) {
			for (int x = 0; x < dimensionSums[0].length; x++) {
				dimensionSums[i][x] = (double) 0;
			}
		}
		
		for(int sampleIter = 0; sampleIter < Driver.sample[0].length; sampleIter++) {//loop thru samples
			for(int dimIter = 0; dimIter < Driver.numInNodes; dimIter++) {//loop thru dimensions of each sample
				
				dimensionSums[labels[sampleIter]][dimIter] += Driver.sample[dimIter][sampleIter];
				labelDivisors[labels[sampleIter]] ++;
			}
		}
		
		for(int labelIter = 0; labelIter < Driver.k; labelIter++) {
			Double[] newCent = new Double[Driver.numInNodes];
			for(int dimIter = 0; dimIter < Driver.numInNodes; dimIter++) {//loop thru dimensions of each sample

				newCent[dimIter] = dimensionSums[labelIter][dimIter] / labelDivisors[labelIter];
			}
			newCentroids.add(labelIter, newCent);
		}
		return newCentroids;
	}

	private static int findMin(Double[] distanceArray) {
		int minIndex = 0;
		for(int index = 1; index < distanceArray.length; index++) {
			if (distanceArray[minIndex] > distanceArray[index]) {
				minIndex = index;
			}
		}
		return minIndex;
	}
	
	private static boolean stopKmeans(ArrayList<Double[]> centroids, ArrayList<Double[]> oldCentroids) {
		
		int index = 0;
		int flags = 0;
		while(centroids.size() > index) {
			
			for(int arrayIter = 0; arrayIter < centroids.get(index).length ; arrayIter++) {
				
				if (centroids.get(index)[arrayIter] - (oldCentroids.get(index)[arrayIter]) < 0.01) {
					flags++;
				}
			}
			index++;
		}

		if (flags >= (Driver.k * Driver.numInNodes * .85)) {
			return true;
		}
		return false;
		
	}
}
