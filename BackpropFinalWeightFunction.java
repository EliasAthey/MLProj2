/**
 * 
 */
package neuralNetScript;

import java.util.ArrayList;
import java.util.ListIterator;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class BackpropFinalWeightFunction implements IWeightFunction {

	private double expectedOutput;

	@Override
	public double computeWeights() {
		// TODO Auto-generated method stub
		return 0;
	}

	//Backpropogation output function
	//Array of weight values and array of data values
	private static double backpropOutFunction(ArrayList<Double> weight, ArrayList<Double> data) throws Exception {
		if (weight.size() != data.size()) {
			throw new Exception("The Backpropogation weight and data arrays are not the same length and thus are not aligned.");
		}
		if (weight.size() < 1 || data.size() < 1) {
			throw new Exception("Backpropogation needs at least one datapoint.");
		}
		double output = -1;
		for (int i = 0; i < data.size() - 1; i++) {
			double sum = 0;
			double datapoint = 0;
			double weightvalue = 0;
			datapoint = data.get(i);
			weightvalue = weight.get(i);
			sum += datapoint*weightvalue; //not quite sure here with the weighted sum 
			output = 1 / (1 + (Math.exp(-1 * (sum))));
		}
		return output;
	}
}
