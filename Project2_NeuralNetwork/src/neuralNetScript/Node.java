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

	// inputs is a 2-by-n matrix, where n is the number of inputs
	// inputs[0][x] contains the x'th input
	// inputs[1][x] contains the weight associated to x'th input
	double[][] inputs;
	
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
		for (int i = 0; i < this.inputs[0].length; i++) {
			sum += (this.inputs[0][i] * this.inputs[1][i]);
		}
		
		this.computedOutput = this.nodeFunction.computeOutput(sum);
	}
	
	// call the weightFunction
	public void updateWeights(){
		// TODO
		this.weightFunction.computeWeights();
	}
	
	// return the set of downstream Nodes
	public Node[] getDownstream(){
		return this.downstream;
	}
	
	// return the delta value used for backprop weight updating
	public double getDeltaValue(){
		return this.deltaValue;
	}
	
	// return the computed output
	public double getComputedOutput(){
		return this.computedOutput;
	}
}
