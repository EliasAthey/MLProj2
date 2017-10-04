/**
 * 
 */
package neuralNetScript;

import java.util.Map;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class Node {
	// interfaces - determines the role of the node
	private INodeFunction nodeFunction;
	private IWeightFunction weightFunction;
	
	// attributes
	Float[][] inputs;
	private Node[] downstream;
	private Float deltaValue;
	private Float computedOutput;
	
	// constructor
	public Node(INodeFunction nodeFunction, IWeightFunction weightFunction, Node[] downstreamNodes){
		this.nodeFunction = nodeFunction;
		this.weightFunction = weightFunction;
		this.downstream = downstreamNodes;
	}
	
	// sum weighted inputs and send to nodeFunction
	public void execute(){
		// sum weighted inputs
		Float sum = null;
		for (int i = 0; i < inputs.length; i++) {
			sum += (inputs[i][0] * inputs[i][1]);
		}
		
		// extra args will be passed to the nodeFunction
		Float[] args = {};
		computedOutput = nodeFunction.computeOutput(sum, args);
	}
	
	// call the weightFunction
	public void updateWeights(){
		// TODO
		Float[] args = {1.0f};// args will be passed to the weightFunction, not sure if we'll need to pass it anything
		weightFunction.computeWeights(args);
	}
	
	// return the set of downstream Nodes
	public Node[] getDownstream(){
		return downstream;
	}
	
	// return the delta value used for backprop weight updating
	public Float getDeltaValue(){
		return deltaValue;
	}
	
	// return the computed output
	public Float getComputedOutput(){
		return computedOutput;
	}
}
