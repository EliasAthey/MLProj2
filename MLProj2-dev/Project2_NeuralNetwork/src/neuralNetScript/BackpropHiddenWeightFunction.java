/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class BackpropHiddenWeightFunction implements IWeightFunction {

	// Ada, the learning rate
	private double learningRate = 1;
	
	@Override
	public void computeWeights(Node node) {
		// determines delta for this node
		double delta = 0;
		double downstreamSum = 0;
		for(Node downstreamNode : node.getDownstream()){
			downstreamSum += downstreamNode.getDeltaValue() * downstreamNode.inputs[1][node.getLayerIndex()];
		}
		delta = -1 * node.getComputedOutput() * (1 - node.getComputedOutput()) * downstreamSum;
		node.setDeltaValue(delta);
		
		// update weights
		for(int i = 0; i < node.inputs[0].length; i++){
			node.inputs[1][i] += this.learningRate * delta * node.inputs[0][i];
		}
	}
}
