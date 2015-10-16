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

public class LeastConfQuerier extends SequenceQuerier {	
	private static DecimalFormat DF = new DecimalFormat("####.######");

	public int[] select(CRF4 model, InstanceList poolData, InstanceList trainData, int num) {
		int[] ret = new int[num];
		Ranker[] scoring = new Ranker[poolData.size()];
		for (int i=0; i<poolData.size(); i++) {
			Instance inst = poolData.getInstance(i);
			Sequence input = (Sequence) inst.getData();
			Transducer.Lattice lattice = model.forwardBackward( input );
			double latticeCost = lattice.getCost();
			Transducer.ViterbiPath vp = model.viterbiPath(input);
			double score = Math.exp ( lattice.getCost() - vp.getCost() );
			scoring[i] = new Ranker(i, score, inst);
		}
		Arrays.sort(scoring);
		for (int i=0; i<ret.length; i++) {
			System.out.println("\t"+toString()+"\t"+scoring[scoring.length-(1+i)]);
			// we actually want to MINIMIZE this score...
			ret[i] = scoring[scoring.length-(1+i)].getIndex();
		}
		Arrays.sort(ret);
		return ret;
	}
	
	/**
	 * toString
	 */
	public String toString() {
		return "LEASTCONF";
	}
}
