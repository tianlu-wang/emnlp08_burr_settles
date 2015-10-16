package emnlp08.pipes;

import edu.umass.cs.mallet.base.types.*;
import edu.umass.cs.mallet.base.pipe.*;
import edu.umass.cs.mallet.base.util.*;
import java.util.*;
import java.io.*;

public class FVS2KernelVector extends Pipe {
	static private boolean wordsOnly = false;

	public FVS2KernelVector () {
		super (Alphabet.class, null);
	}
	public FVS2KernelVector (boolean wordsOnly) {
		super (Alphabet.class, null);
		this.wordsOnly = wordsOnly;
	}
	
	public Instance pipe (Instance carrier) {
		HashMap hash = new HashMap();
//		SparseVector sv = new SparseVector();
		FeatureVectorSequence fvs = (FeatureVectorSequence)carrier.getData();
		String[] tokens = carrier.getName().toString().split("[\t ]+");
		for (int i = 0; i < fvs.size(); i++) {
			FeatureVector fv = (FeatureVector)fvs.get(i);
			int[] indices = fv.getIndices();
//			System.out.println(indices.length+" // "+tokens[i]+" // "+fv.toString(true));
			for (int j=0; j<indices.length; j++) {
				int idx = indices[j];
				String sidx = ""+idx;
				double val = fv.value(idx);
				if (hash.containsKey(new Integer(indices[j]))) {
					hash.put(sidx, new Double(val + ((Double)hash.get(sidx)).doubleValue()));
				} else {
					hash.put(sidx, new Double(val));
				}
			}
		}
		// done. now dump these features into a sparse vector
		int[] indices = new int[hash.size()];
		double[] values = new double[hash.size()];
		Iterator it = hash.keySet().iterator();
		int ki = 0;
		while (it.hasNext()) {
			String sidx = (String)it.next();
			indices[ki] = Integer.parseInt(sidx);
			values[ki] = ((Double)hash.get(sidx)).doubleValue();
			ki++;
		}
		// shove the sparse vector into source!
		SparseVector sv = new SparseVector(indices, values);
		carrier.setSource(sv);
		return carrier;
	}
}
