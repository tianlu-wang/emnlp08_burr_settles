package emnlp08.pipes;

import edu.umass.cs.mallet.base.types.*;
import edu.umass.cs.mallet.base.pipe.*;
import edu.umass.cs.mallet.base.util.*;
import java.util.*;
import java.io.*;

public class TokenSequence2SparseVector extends Pipe {
	static private boolean wordsOnly = false;

	public TokenSequence2SparseVector () {
		super (Alphabet.class, null);
	}
	public TokenSequence2SparseVector (boolean wordsOnly) {
		super (Alphabet.class, null);
		this.wordsOnly = wordsOnly;
	}
	
	public Instance pipe (Instance carrier) {
		Alphabet alph = (Alphabet)getDataAlphabet();
		int[] buffer = new int[alph.size()];
		Arrays.fill(buffer, 0);
		TokenSequence ts = (TokenSequence) carrier.getData();
		for (int i = 0; i < ts.size(); i++) {
			PropertyList pl = ts.getToken(i).getFeatures();
			if (pl != null) {
				PropertyList.Iterator iter = pl.iterator();
				while (iter.hasNext()) {
					iter.next();
					String feature = (String)iter.getKey();
					double v = iter.getNumericValue();
					int idx = alph.lookupIndex(iter.getKey(), false);
					if (idx >= 0) {
						if (!wordsOnly || (feature.startsWith("W=") && !feature.contains("@"))) {
//							System.out.print(buffer.length+" // "+idx);
//							System.out.println(" --> "+feature+'='+(buffer[idx]+1));
							buffer[idx]++;
						}
					}
/*					if (v == 1.0)
						sb.append (iter.getKey());
					else
						sb.append (iter.getKey()+'='+v);
					sb.append (' ');*/
				}
			}
		}
		// done. now dump these features into a sparse vector
		ArrayList tmpIdx = new ArrayList();
		ArrayList tmpVal = new ArrayList();
		for (int i=0; i<buffer.length; i++) {
			if (buffer[i] != 0.0 && buffer[i] != Double.NEGATIVE_INFINITY && buffer[i] != Double.POSITIVE_INFINITY) {
				tmpIdx.add(new Integer(i));
				tmpVal.add(new Double(buffer[i]));
			}
		}
		assert(tmpIdx.size() == tmpVal.size());
		int[] indices = new int[tmpIdx.size()];
		double[] values = new double[tmpVal.size()];
		for (int i=0; i<tmpIdx.size(); i++) {
			indices[i] = ((Integer)tmpIdx.get(i)).intValue();
			values[i] = Math.abs(((Double)tmpVal.get(i)).doubleValue());
		}
		// shove the sparse vector into source!
		SparseVector sv = new SparseVector(indices, values);
		carrier.setSource(sv);
		return carrier;
	}
}
