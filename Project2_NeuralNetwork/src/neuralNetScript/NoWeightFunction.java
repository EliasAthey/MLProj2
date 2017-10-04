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
	public Float computeWeights() {
		//if this class is just going to return the sum of args[] we can just use the 
		//RadialBasisOutFunction
		return null;
	}

}
