/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class Node {
	// interfaces - determines the role of the node
	private INodeFunction nodeFunction;
	private IWeightFunction weightFunction;
	
	// attributes
	private Node[] downstream;
	private double deltaValue;
	private double computedOutput;

	// first dimension contains all the input values
	// second dimension contains their associated weights
	Float[][] inputs;
	
	// constructor
	public Node(INodeFunction nodeFunction, IWeightFunction weightFunction, Node[] downstreamNodes){
		this.nodeFunction = nodeFunction;
		this.weightFunction = weightFunction;
		this.downstream = downstreamNodes;
	}
	
	// sum weighted inputs and send to nodeFunction
	public void execute(){
		// sum weighted inputs
		double sum = 0;
		for (int i = 0; i < inputs.length; i++) {
			sum += (inputs[i][0] * inputs[i][1]);
		}
		
		computedOutput = nodeFunction.computeOutput(sum);
	}
	
	// call the weightFunction
	public void updateWeights(){
		// TODO
		weightFunction.computeWeights();
	}
	
	// return the set of downstream Nodes
	public Node[] getDownstream(){
		return downstream;
	}
	
	// return the delta value used for backprop weight updating
	public double getDeltaValue(){
		return deltaValue;
	}
	
	// return the computed output
	public double getComputedOutput(){
		return computedOutput;
	}
}
