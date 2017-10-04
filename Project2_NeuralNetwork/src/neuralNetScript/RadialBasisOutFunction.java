/**
 * 
 */
package neuralNetScript;
/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class RadialBasisOutFunction implements INodeFunction {

	@Override
	public Float computeOutput(Float args[]) {
		Float sum = null;
		for (int i = 0; i < args.length; i++) {
			sum += args[i];
		}
		return sum;
	}

}
