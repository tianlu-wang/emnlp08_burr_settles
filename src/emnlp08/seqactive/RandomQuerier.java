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

public class RandomQuerier extends SequenceQuerier {	
	private static DecimalFormat DF = new DecimalFormat("####.######");

	public int[] select(CRF4 model, InstanceList poolData, InstanceList trainData, int num) {
		Random r = new Random();
		int[] ret = new int[num];
		for (int i=0; i<ret.length; i++) {
			ret[i] = r.nextInt(poolData.size());
			System.out.println("\t"+toString()+"\t"+ret[i]+" : "+((Instance)poolData.get(ret[i])).getName());
		}
		Arrays.sort(ret);
		return ret;
	}

	/**
	 * toString
	 */
	public String toString() {
		return "RANDOM";
	}
}
