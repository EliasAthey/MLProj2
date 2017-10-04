/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class NoFunction implements INodeFunction {

	@Override
	public Float computeOutput(Float args[]) {
		//if this class is just going to return the sum of args[] we can just use the 
		//RadialBasisOutFunction
		return null;
	}

}
