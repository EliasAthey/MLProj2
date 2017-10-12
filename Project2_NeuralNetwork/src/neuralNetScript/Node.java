/**
 * 
 */
package neuralNetScript;

import java.lang.reflect.InvocationTargetException;

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
	private int layerIndex;

	// inputs is a 2-by-n matrix, where n is the number of inputs
	// inputs[0][x] contains the x'th input
	// inputs[1][x] contains the weight associated to x'th input
	double[][]inputs;
	
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
		for(Node node : this.downstream){
			node.inputs[0][this.layerIndex] = this.computedOutput;
		}
	}
	
	// call the weightFunction
	public void updateWeights(){
		this.weightFunction.computeWeights(this);
	}
	
	// return the set of downstream Nodes
	public Node[] getDownstream(){
		return this.downstream;
	}
	
	// set the delta value
	public void setDeltaValue(double delta){
		this.deltaValue = delta;
	}
	
	// return the delta value used for backprop weight updating
	public double getDeltaValue(){
		return this.deltaValue;
	}
	
	// return the computed output
	public double getComputedOutput(){
		return this.computedOutput;
	}
	
	// set the layer index
	public void setLayerIndex(int index){
		this.layerIndex = index;
	}
	
	// get the layer index
	public int getLayerIndex(){
		return this.layerIndex;
	}
	
	// set the layer index
	public void setAssociatedCluster(Double[] cluster){
		if(this.nodeFunction.getClass().getTypeName().equals("RadialBasisFunction")){
			try{
				this.nodeFunction.getClass().getMethod("setAssociatedCluster", double[].class).invoke(this.nodeFunction, cluster);
			}
			catch(NoSuchMethodException e){
				System.out.println(e.getMessage());
			} 
			catch (IllegalAccessException e) {
				System.out.println(e.getMessage());
			} 
			catch (InvocationTargetException e) {
				System.out.println(e.getMessage());
			}
		}
	}
}
