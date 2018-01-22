package cse417;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import java.util.function.DoubleFunction;


/**
 * Program that finds {@code TeamModel}s that best fit provided data. The
 * option {@code --train} shows the results of using different penalties when
 * fitting historical data. Without that option, this just finds the best model
 * using the last {@code WEEKS} weeks of data. The latter supports using either
 * a fixed penalty or finding the model with a fixed number of non-zero params.
 */
public class TeamModeler {

	/** Number of weeks to use for building models. */
	private static int WEEKS = 6;

	/**
	 * Tolerance parameter used by coordinate descent. The method will stop
	 * iterating when the L0 norm of the change is less than this amount.
	 */
	private static double TOLERANCE = 5e-3;

	/** Entry point for a program to build a model of NFL teams. */
	public static void main(String[] args) throws IOException {
		ArgParser argParser = new ArgParser("TeamModeler");
		argParser.addOption("train", Boolean.class);
		argParser.addOption("penalty", Double.class);
		argParser.addOption("num-nonzero", Integer.class);
		argParser.addOption("verbose", Boolean.class);
		args = argParser.parseArgs(args, 1, 1);

		if (argParser.hasOption("train")) {
			train(args[0]);
		} else {
			// Find the drives in the last WEEKS worth of weeks.
			List<Drive> drives = loadDrives(args[0], 1, 16);
			int maxWeek = drives.stream().map((d) -> d.week)
					.max((a, b) -> Double.compare(a, b)).get();
			drives = drives.stream().filter((d) -> maxWeek - WEEKS + 1 <= d.week)
					.collect(Collectors.toList());

			TeamModel model;
			if (argParser.hasOption("num-nonzero")) {
				model = findBestSparseModel(drives,
						argParser.getIntegerOption("num-nonzero"), TOLERANCE,
						argParser.hasOption("verbose"));
			} else {
				double penalty = argParser.hasOption("penalty") ?
						argParser.getDoubleOption("penalty") : 0.0;
						model = findBestModel(drives, penalty, TOLERANCE,
								argParser.hasOption("verbose"));
			}
			model.printTo(System.out);
		}
	}

	/**
	 * Returns the drives described in the given file from games in weeks of the
	 * NFL season in the given range ({@code minWeek} to {@code maxWeek}).
	 */
	private static List<Drive> loadDrives(
			String fileName, int minWeek, int maxWeek) throws IOException {
		List<Drive> drives = new ArrayList<Drive>();
		CsvParser parser = new CsvParser(fileName, true, new Object[] {
				String.class, String.class, Integer.class, Float.class, Float.class
		});
		while (parser.hasNext()) {
			String[] parts = parser.next();
			int week = Integer.parseInt(parts[2]);
			if (minWeek <= week && week <= maxWeek) {
				double expStartPts = Double.parseDouble(parts[3]);
				double expEndPts = Double.parseDouble(parts[4]);
				drives.add(new Drive(week, parts[0], parts[1], expStartPts, expEndPts));
			}
		}
		return drives;
	}

	/**
	 * Fills in the given maps with the best model of how the offenses increase
	 * the expected points in each drive and how the defenses decrease it.
	 * @param drives List of the drives that the model should describe.
	 * @param penalty Constant factor on the penalty term of the loss function.
	 * @param tol Stop when the L0 change per iteration less than this amount.
	 * @param verbose If true, prints progress of the model fitting process.
	 */
	private static TeamModel findBestModel(
			final List<Drive> drives, double penalty, double tol, boolean verbose) {
		// TODO: implement coordinate descent
		TeamModel best = new TeamModel();
		double change = Double.MAX_VALUE;
		List<String> teams = TeamModel.TEAMS;
		while (change > tol) {
			TeamModel temp = best.copy();
			for (String team: teams) {
				double newOffense = Optimizer.findMinimumOfUnimodal(t -> temp.copy().setOffense(team, t).evalLoss(drives, penalty), -2, 6);
				temp.setOffense(team, newOffense);
				double newDefense = Optimizer.findMinimumOfUnimodal(t -> temp.copy().setDefense(team, t).evalLoss(drives, penalty), -2, 6);
				temp.setDefense(team, newDefense);
			}
			double newConstant = Optimizer.findMinimumOfUnimodal(t -> temp.copy().setConstant(t).evalLoss(drives, penalty), -8, 8);
			temp.setConstant(newConstant);
			change = best.copy().addScaledBy(-1, temp).norm0();
			best = temp;
		}
		return best;
	}

	/**
	 * Like {@code findBestModel} above but taking the desired number of non-zero
	 * parameters in the model instead of the penalty.
	 * <p>
	 * If decreasing the penalty by less than 0.0001 changes the number of
	 * non-zero parameters from less than the desired amount to more than the
	 * desired amount, then this will just return the model with fewer parameters
	 */
	// Invariant: Invariant: (f(p) < x for p <= low) and (x < f(p) or p >= high) 
	private static TeamModel findBestSparseModel(
			final List<Drive> drives, int numNonZeros, double tol, boolean verbose) {
		// TODO: Extra credit: implement this by figuring out which penalty to use
		//       to get the desired number of non-zero entries
		TeamModel best = new TeamModel();
		int currCount = 0;
		double low = 0.0;
		double high = 0.5;
		double penalty = 0.0;
		while (currCount != numNonZeros) {
			TeamModel temp = new TeamModel();
			double res = (low + high) / 2;
			List<String> teams = TeamModel.TEAMS;
			for (String team: teams) {
				double newOffense = Optimizer.findMinimumOfUnimodal(t -> temp.copy().setOffense(team, t).evalLoss(drives, penalty), -2, 6);
				temp.setOffense(team, newOffense);
			}
			for (String team: teams) {
				double newDefense = Optimizer.findMinimumOfUnimodal(t -> temp.copy().setDefense(team, t).evalLoss(drives, penalty), -2, 6);
				temp.setOffense(team, newDefense);
			}
			double newConstant = Optimizer.findMinimumOfUnimodal(t -> temp.copy().setConstant(t).evalLoss(drives, penalty), -8, 8);
			temp.setConstant(newConstant);
			currCount = temp.countNonZeroParameters(tol);
			if (currCount < numNonZeros) {
				high = penalty;
			} else if (currCount > numNonZeros) {
				low = penalty;
			} else {
				res = 0.0;
				best = temp.copy();
			}
		}
		return best;
	}

	/** 
	 * Prints the results of using different penalties .001 to 0.050 on each
	 * period of {@code WEEKS} weeks in the provided data. That is, for each
	 * period of {@code WEEKS+1} consecutive weeks in the data, it trains a model
	 * using the first {@code WEEKS} of data (for each penalty) and then shows
	 * the results for predicting the next week.
	 * <p>
	 * <b>Important</b>: Errors are normalized by subtracting off the minimum
	 * error for any of the penalties tried for those weeks. This means, for each
	 * period of {@code WEEKS} weeks, at least one penalty gives error zero.
	 * (This normalization is necessary because some weeks are harder to predict
	 * than others, and we want the model that works best on an average week not
	 * just the worst week.)
	 */
	private static void train(String fileName) throws IOException {
		// TODO: try every period of WEEKS weeks in a row
		int min = 1;
		int max = WEEKS + 1;
		while (max < 17) {
			List<Integer> nonZeroParams = new ArrayList<Integer>();
			List<Double> testErrors = new ArrayList<Double>();
			List<Drive> drives = loadDrives(fileName, min, max - 1);
			List<Drive> next = loadDrives(fileName, max, max);
			double minTestError = Double.MAX_VALUE;
			for (double pc = 0.050; pc > -0.001; pc -= 0.001) {
				TeamModel model = findBestModel(drives, pc, TOLERANCE, true);
				double testError = model.evalLoss(next, 0.0);
				minTestError = Math.min(minTestError, testError);
				nonZeroParams.add(model.countNonZeroParameters(TOLERANCE));
				testErrors.add(testError);
			}
			for (int i = 0; i < testErrors.size(); i++) {
				System.out.printf("%2d %g", nonZeroParams.get(i), (testErrors.get(i) - minTestError));
				System.out.println();
			}
			min++;
			max++;
		}
	}
}
