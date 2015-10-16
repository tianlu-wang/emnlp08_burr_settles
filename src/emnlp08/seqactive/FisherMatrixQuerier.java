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

public class FisherMatrixQuerier extends SequenceQuerier {	
	private static DecimalFormat DF = new DecimalFormat("####.######");
	private static int ENT_WINDOW = 20;

	public FisherMatrixQuerier() {}
	public FisherMatrixQuerier(int k) {
		ENT_WINDOW = k;
	}

	public int[] select(CRF4 model, InstanceList poolData, InstanceList trainData, int num) {
		int[] ret = new int[num];
		double[] scores = new double[poolData.size()];
		Arrays.fill(scores, 0.0);
		double[] poolMatrix = new double[model.numParameters(trainData)];
		Arrays.fill(scores, 0.0);
		SparseVector[] fims = new SparseVector[poolData.size()];
		// first, calculate individual instance Fisher matrices, and build overall FIM
		for (int i=0; i<poolData.size(); i++) {
			Instance inst = (Instance)poolData.get(i);
			fims[i] = fisherMatrix(model, inst, poolMatrix.length, poolData.getTargetAlphabet());
			// add this FIM's contribution to the pool's FIM
			int[] indices = fims[i].getIndices();
			for (int ii=0; ii<indices.length; ii++) {
				int k = indices[ii];
				poolMatrix[k] += fims[i].value(k) / poolData.size();
			}
		}
		// second, compute the ratios between them for scoring
		for (int i=0; i<poolData.size(); i++) {
			for (int k=0; k<poolMatrix.length; k++)
				// only worry if this is nonzero in pool FIM
				if (poolMatrix[k] > 0)
					scores[i] += (poolMatrix[k] + 1e-7) / (fims[i].value(k) + 1e-7);
		}
		// now build the soring list
		Ranker[] scoring = new Ranker[poolData.size()];
		for (int i=0; i<scores.length; i++) {
			Instance inst = (Instance)poolData.get(i);
			scoring[i] = new Ranker(i, scores[i], inst);
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
	 * fisherMatrix
	 */
	public static SparseVector fisherMatrix(CRF4 model, Instance inst, int numParams, Alphabet targetAlphabet) {
		Sequence input = (Sequence) inst.getData();
		// (1) calculate the "lattice cost":
		Transducer.Lattice lattice = model.forwardBackward(input);
		double latticeCost = lattice.getCost();
		// (2) calculate the costs for the top ENT_WINDOW sequences:
		Transducer.ViterbiPath_NBest vpnb = model.viterbiPath_NBest ( input, ENT_WINDOW );
		double[] costs = vpnb.costNBest();
		Sequence[] labels = vpnb.outputNBest();
		double[] probs = new double[costs.length];
		// (3) normalize+exponentiate for each, get margin
		double totalProb = 0;
		for (int j=0; j<costs.length; j++) {
			probs[j] = Math.exp( latticeCost - costs[j] );
			totalProb += probs[j];
		}
		// build a dense vector for the expected diagonal of FIM
		double[] buffer = new double[numParams];
		Arrays.fill(buffer, 0.0);
		for (int j=0; j<probs.length; j++) {
			double p = probs[j] / totalProb;
			if (p > 0) {
				// fabricate instance with this labeling
				Instance tmpInst = (Instance)inst.shallowCopy();
				tmpInst.unLock();				
//				System.out.println(DF.format(p)+"\t"+labels[j]);
				tmpInst.setTarget(makeLabelSequence(labels[j], targetAlphabet));
				// fabricate list w/ fake instance
				InstanceList tmpList = new InstanceList();
				tmpList.add(tmpInst);
				SparseVector tmpSV = model.getGradient(tmpList);

//				System.out.println("###################################\n"+inst.getName());
//				System.out.println("\t"+tmpSV.toString(true));

				int[] tmpIdx = tmpSV.getIndices();
				for (int ii=0; ii<tmpIdx.length; ii++) {
					int k = tmpIdx[ii];
					// make sure this is a feature we've even seen before...
					if (k < buffer.length)
						buffer[k] += p * Math.pow(tmpSV.value(k), 2);
				}
			}
		}
		// convert that dense buffer to a sparse vector
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
		SparseVector ret = new SparseVector(indices, values);
//		System.out.println("###################################\n"+inst.getName());
//		System.out.println(input.size()+"\t"+DF.format(ret.twoNorm()));
		return ret;
	}

	/**
	 * makeLabelSequence
	 */
	public static LabelSequence makeLabelSequence(Sequence s, Alphabet targetAlphabet) {
		LabelSequence target = new LabelSequence ((LabelAlphabet)targetAlphabet, s.size());
		for (int i=0; i<s.size(); i++) {
			target.add(s.get(i).toString());
		}
		return target;
	}
	
	/**
	 * toString
	 */
	public String toString() {
		return "FISHER";
	}
	
}
