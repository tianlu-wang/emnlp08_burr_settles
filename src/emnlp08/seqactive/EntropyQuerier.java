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

public class EntropyQuerier extends SequenceQuerier {	
	private static DecimalFormat DF = new DecimalFormat("####.######");
	private static int ENT_WINDOW = 0;

	public EntropyQuerier(){}
	public EntropyQuerier(int k) {
		ENT_WINDOW = k;
	}

	public int[] select(CRF4 model, InstanceList poolData, InstanceList trainData, int num) {
		int[] ret = new int[num];
		Ranker[] scoring = new Ranker[poolData.size()];
		for (int i=0; i<poolData.size(); i++) {
			Instance inst = (Instance)poolData.get(i);
			double score = (ENT_WINDOW > 1)
				? Misc.kBestSequenceEntropy(model, inst, ENT_WINDOW)
				: Misc.sequenceEntropy(model, inst);
			scoring[i] = new Ranker(i, score, inst);
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
		return "ENTROPY"+((ENT_WINDOW > 1) ? "("+ENT_WINDOW+")" : "");
	}
}
