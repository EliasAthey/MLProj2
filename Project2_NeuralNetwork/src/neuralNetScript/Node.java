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
	Map<Float, Float> inputs;//Key: input value, Value: input weight
	private Node[] downstream;
	private Float deltaValue;
	private Float computedOutput;
	
	// collect weighted inputs and send to nodeFunction
	public void execute(){
		// TODO
		Float[] args = {1.0f};// args will be passed to the nodeFunction
		computedOutput = nodeFunction.computeOutput(args);
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
