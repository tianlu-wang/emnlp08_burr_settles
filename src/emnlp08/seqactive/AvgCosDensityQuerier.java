package emnlp08.seqactive;

import emnlp08.*;
import edu.umass.cs.mallet.base.classify.*;
import edu.umass.cs.mallet.base.types.*;
import edu.umass.cs.mallet.base.fst.*;
import edu.umass.cs.mallet.base.fst.confidence.*;
import edu.umass.cs.mallet.base.minimize.*;
import edu.umass.cs.mallet.base.minimize.tests.*;
import edu.umass.cs.mallet.base.pipe.*;
import edu.umass.cs.mallet.base.pipe.iterator.*;
import edu.umass.cs.mallet.base.pipe.tsf.*;
import edu.umass.cs.mallet.base.util.CharSequenceLexer;
import junit.framework.*;
import java.util.*;
import java.util.regex.*;
import java.io.*;
import java.text.DecimalFormat;

public class AvgCosDensityQuerier extends SequenceQuerier {	
	private static DecimalFormat DF = new DecimalFormat("####.######");
	public static Metric dpMetric = new NormalizedDotProductMetric();

	public int[] select(CRF4 model, InstanceList poolData, InstanceList trainData, int num) {
		int[] ret = new int[num];
		double[] scores = new double[poolData.size()];
		Arrays.fill(scores, 0.0);
		// compute what the kernel vectors look like first
		Misc.makeKernelVectors(poolData, trainData.getFeatureSelection());
		// second, compute the "average" vector based on all instances
		SparseVector mv = Misc.makeAverageVector(poolData);
		// lastly, cycle through and "density-weight" the entropies...
		for (int i=0; i<poolData.size(); i++) {
			Instance inst = (Instance)poolData.get(i);
			double ent = Misc.sequenceEntropy(model, inst);
			double dens = 1-dpMetric.distance(mv, (SparseVector)inst.getSource());
			scores[i] = ent * dens;
		}
		// now build the soring list
		Ranker[] scoring = new Ranker[poolData.size()];
		for (int i=0; i<scores.length; i++) {
			Instance inst = (Instance)poolData.get(i);
			scoring[i] = new Ranker(i, scores[i], inst);
		}
		Arrays.sort(scoring);
		for (int i=0; i<ret.length; i++) {
			System.out.println("\t"+toString()+"\t"+scoring[i]);
			ret[i] = scoring[i].getIndex();
		}
		Arrays.sort(ret);
		return ret;
	}

	/**
	 * toString
	 */
	public String toString() {
		return "AVGCOS";
	}
}
