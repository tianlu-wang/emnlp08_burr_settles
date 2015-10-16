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

// this computes the average entropy of tokens in a sequece...

public class TokenEntropyQuerier extends SequenceQuerier {	
	private static DecimalFormat DF = new DecimalFormat("####.######");
	private static int ENT_WINDOW = 0;
	private static boolean AVERAGE = false;
	
	public TokenEntropyQuerier() {
		this(false);
	}
	public TokenEntropyQuerier(boolean avg) {
		AVERAGE = avg;
	}

	public int[] select(CRF4 model, InstanceList poolData, InstanceList trainData, int num) {
		int[] ret = new int[num];
		Ranker[] scoring = new Ranker[poolData.size()];
		for (int i=0; i<poolData.size(); i++) {
			double score = 0;
			Instance inst = (Instance)poolData.get(i);
		    Sequence input = (Sequence) inst.getData();
		    Transducer.Lattice lattice = model.forwardBackward (input, null, false, (LabelAlphabet)poolData.getTargetAlphabet());
		    for (int j = 0; j < input.size(); j++) {
				LabelVector lv = lattice.getLabelingAtPosition(j);
				double tokenEntropy = Misc.entropy(lv);
				score += tokenEntropy;
		    }
			if (AVERAGE)
		    	score /= input.size();
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
		return "TOKENENT";
	}	
}
