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
	// the type of network being implemented
	private static String networkType;
	
	// the number of inputs
	private static int numInNodes;
	
	// an array list containing the number of nodes in each hidden layer. size() is number of hidden layers
	private static ArrayList<Integer> numHiddenLayers = new ArrayList<Integer>();
	
	// the number of output nodes
	private static int numOutNodes;
	
	// the convergence time
	private static double convergenceTime;
	
	// the weights of the network ont he previous iteration of updating weights
	private static ArrayList<Double> prevWeights;
	
	// the training sample
	private static Double[][] sample;
	
	// the k-value used for k-means clustering
	private static int k;
	
	// the sigma value used for rbf
	private static final double sigma = 100;
	
	// the network itself
	private static ArrayList<Layer> network;
	
	// package accessible sample expected output
	static double expectedOutput;
	
	// the entry point of the program
	//args[0] = networkType
	//args[1] = layers
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
		catch(Exception e){
			System.out.println("Error...");
			System.out.println(e.getMessage());
			for(StackTraceElement stack : e.getStackTrace()){
				System.out.println(stack);
			}
		}
		
		// test the network with a random set of samples sample
		Double[][] testSample = Driver.getSample((int)Math.pow(1.8, Driver.numInNodes) * 100);
		double[] computedOutput = new double[Driver.numOutNodes];
		for(int datapoint = 0; datapoint < testSample[0].length; datapoint++){
			Double[] input = new Double[testSample.length - 1];
//			System.out.print("\nNetwork test on {");
			for(int i = 0; i < testSample.length - 2; i++){
//				System.out.print(testSample[i][datapoint] + ", ");
				input[i] = testSample[i][datapoint];
			}
//			System.out.print(testSample[testSample.length - 2][datapoint] + "}");
			input[testSample.length - 2] = testSample[testSample.length - 2][datapoint];
			computedOutput = Driver.testNetwork(input);
//			System.out.print(": " + computedOutput);
//			System.out.println("\nActual Rosenbrock value: " + testSample[testSample.length - 1][datapoint]);
		}
		
		// print some metrics
		double avgError = 0.0;
		for(int i = 0; i < computedOutput.length; i++){
			avgError += Math.abs(computedOutput[i] - testSample[testSample.length - 1][i]);
		}
		avgError = avgError / computedOutput.length;
		System.out.println("Average Error in Output: " + avgError);
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
				inputs.add(inputIter, Math.random() * 1000);
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
		
		// set k-value, clusters, and sigma for rbf network
		ArrayList<Double[]> clusters = new ArrayList<Double[]>();
		if(Driver.networkType.equals("rbf")){
			Driver.k = Driver.numHiddenLayers.get(0);
			clusters = kmeans();
			RadialBasisFunction.setSigma(Driver.sigma);
		}
		
		// create hidden layer nodes and store in hidden layer
		Node[] prevHiddenNodes = outputNodes;
		for(int i = hiddenLayers.length - 1; i >= 0; i--){
			Node[] hiddenNodes = new Node[Driver.numHiddenLayers.get(i)];
			for(int j = 0; j < hiddenNodes.length; j++){
				// set the node functions for hidden nodes
				if(i == hiddenLayers.length - 1){
					switch(Driver.networkType){
						case "rbf": 
							hiddenNodes[j] = new Node(new RadialBasisFunction(), new NoWeightFunction(), outputNodes);
							// set the associated cluster
							hiddenNodes[j].setAssociatedCluster(clusters.get(j));
							break;
						case "mlp":
							hiddenNodes[j] = new Node(new SigmoidalFunction(), new BackpropHiddenWeightFunction(), outputNodes);
							break;
					}
				}
				else{
					if(Driver.networkType.equals("rbf")){
						throw new Exception("An rbf network cannot have more than one hidden layer.");
					}
					hiddenNodes[j] = new Node(new SigmoidalFunction(), new BackpropHiddenWeightFunction(), prevHiddenNodes);
				}
				hiddenNodes[j].setLayerIndex(j);
			}
			
			// initialize input arrays with random weights between -0.5 and 0.5 for downstream nodes
			for(int m = 0; m < prevHiddenNodes.length; m++){
				prevHiddenNodes[m].inputs = new double[2][hiddenNodes.length];
				for(int k = 0; k < prevHiddenNodes[m].inputs[1].length; k++){
					int sign = 1;
					if(Math.random() < 0.5){
						sign = -1;
					}
					prevHiddenNodes[m].inputs[1][k] = sign * (Math.random() * 0.5);
				}
			}
			
			// set hidden nodes to this layer, set reference to these hidden nodes
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
			
			// initialize input node weights with 1
			inputNodes[i].inputs = new double[2][1];
			inputNodes[i].inputs[1][0] = 1;
			inputNodes[i].setLayerIndex(i);
		}
		
		// initialize first hidden layer (or output layer in the case with no hidden layers) input arrays with random weights
		for(int j = 0; j < prevHiddenNodes.length; j++){
			prevHiddenNodes[j].inputs = new double[2][inputNodes.length];
			for(int k = 0; k < prevHiddenNodes[j].inputs[1].length; k++){
				switch(Driver.networkType){
					case "rbf": 
						prevHiddenNodes[j].inputs[1][k] = 1.0;
						break;
					case "mlp":
						int sign = 1;
						if(Math.random() < 0.5){
							sign = -1;
						}
						prevHiddenNodes[j].inputs[1][k] = sign * (Math.random() * 0.5);
						break;
				}
			}
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
			System.out.println("Sample Iteration: " + i);
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
			Driver.prevWeights = new ArrayList<Double>();
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
			if(!hasConverged){
				break;
			}
			hasConverged &= (int)(allWeights.get(i) * 1000) == (int)(Driver.prevWeights.get(i) * 1000);
		}
		return hasConverged;
	}
	
	// given an input vector, return the output of the network as the approximation of the Rosenbrock function
	private static double[] testNetwork(Double[] input){
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
		Double[] distances = new Double[Driver.k];
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

	// determines the minimum value in the distance array
	private static int findMin(Double[] distanceArray) {
		int minIndex = 0;
		for(int index = 1; index < distanceArray.length; index++) {
			if (distanceArray[minIndex] > distanceArray[index]) {
				minIndex = index;
			}
		}
		return minIndex;
	}
	
	// a convergence test for the k-means algorithm that returns true if the cluster vectors have converged
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
