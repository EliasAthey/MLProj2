/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class NoWeightFunction implements IWeightFunction {

	@Override
	public Float computeWeights(Float[] args) {
		//if this class is just going to return the sum of args[] we can just use the 
		//RadialBasisOutFunction
		return null;
	}

}
