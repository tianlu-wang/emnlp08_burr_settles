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

// queries based on largest average KL divergence from consensus, based on top K parses of each committee member
// QBC done using query-by-bagging method

public class CommitteeQuerier extends SequenceQuerier {	
	private static DecimalFormat DF = new DecimalFormat("####.######");
	private static int COMMITTEE_SIZE = 3;
	private static int PER_BATCH_SIZE = 15;
	private static boolean VOTE_ENTROPY = false;
/*	private static CRF4[] committee = null;*/
	
	public CommitteeQuerier() {
		this(COMMITTEE_SIZE, PER_BATCH_SIZE, VOTE_ENTROPY);
	}
	public CommitteeQuerier(boolean voteEntropy) {
		this(COMMITTEE_SIZE, PER_BATCH_SIZE, voteEntropy);
	}
	public CommitteeQuerier(int committeeSize, boolean voteEntropy) {
		this(committeeSize, PER_BATCH_SIZE, voteEntropy);
	}
	public CommitteeQuerier(int committeeSize) {
		this(committeeSize, PER_BATCH_SIZE, VOTE_ENTROPY);
	}
	public CommitteeQuerier(int committeeSize, int batchSize, boolean voteEntropy) {
		COMMITTEE_SIZE = committeeSize;
		PER_BATCH_SIZE = batchSize;
		VOTE_ENTROPY = voteEntropy;
/*		CRF4[] committee = new CRF4[COMMITTEE_SIZE];*/
	}

	public int[] select(CRF4 model, InstanceList poolData, InstanceList trainData, int num) {
		int[] ret = new int[num];
		CRF4[] committee = new CRF4[COMMITTEE_SIZE];
		InstanceList[] trainSplits = new InstanceList[COMMITTEE_SIZE];
		// create training sets and learn models
		for (int i=0; i<COMMITTEE_SIZE; i++) {
			System.out.print("QBC("+i+")...");
			trainSplits[i] = trainData.sampleWithReplacement(new Random(), trainData.size());
			committee[i] = new CRF4(model); // just copy the current model for starters... // model.getInputPipe(), null);
//			committee[i].addStatesForLabelsConnectedAsIn(poolData);
			committee[i].train(trainSplits[i], null, null, null, 100);
		}
		System.out.println();
		// prepare to score each instance...
		Ranker[] scoring = new Ranker[poolData.size()];
		double fubar = 0;
		for (int i=0; i<poolData.size(); i++) {
			Instance inst = (Instance)poolData.get(i);
			Sequence[] labelings = getLabelings(committee, inst);
			fubar += 1.0*labelings.length/poolData.size();
			double[][] probs = getProbs(committee, inst, labelings);
			double[] avgProbs = getAvgProbs(probs);
			double score = 0;
			if (VOTE_ENTROPY)
				// use "vote entropy" for QBC scoring...
				score = Misc.entropy(avgProbs);
			else {
				// or, use average KL divergence (default)...
				for (int j=0; j<COMMITTEE_SIZE; j++)
					score += Misc.divergenceKL(probs[j], avgProbs)/COMMITTEE_SIZE;				
			}
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
	 * getLabelings
	 */
	public static Sequence[] getLabelings(CRF4[] committee, Instance inst) {
		HashMap hash = new HashMap();
		Sequence input = (Sequence) inst.getData();
		for (int i=0; i<COMMITTEE_SIZE; i++) {
			Transducer.ViterbiPath_NBest vpnb = committee[i].viterbiPath_NBest ( input, PER_BATCH_SIZE );
			Sequence[] labels = vpnb.outputNBest();
			for (int j=0; j<labels.length; j++)
				hash.put(labels[j].toString(), labels[j]);
		}
		
		Collection set = hash.values();
		Sequence[] ret = (Sequence[])set.toArray(new Sequence[0]);
		
		if (false && ret.length != 15) {
			System.out.println("\n\n# of labelings: "+ret.length+"\n\n");
			for (int i=0; i<ret.length; i++)
				System.out.println(ret[i]);			
		}
		
		return ret;
	}
	
	/**
	 * getProbs
	 */
	public static double[][] getProbs(CRF4[] committee, Instance inst, Sequence[] labelings) {
		Sequence input = (Sequence) inst.getData();
		double[][] ret = new double[COMMITTEE_SIZE][labelings.length];
		double[][] probs = new double[COMMITTEE_SIZE][labelings.length];
		double[] norms = new double[COMMITTEE_SIZE];
		Arrays.fill(norms, 0.0);
		for (int i=0; i<COMMITTEE_SIZE; i++) {
			Transducer.Lattice fullLattice = committee[i].forwardBackward(input);
			double fullCost = fullLattice.getCost();			
			for (int j=0; j<labelings.length; j++) {
				Transducer.Lattice lattice = committee[i].forwardBackward(input, labelings[j]);
				probs[i][j] = Math.exp( fullCost - lattice.getCost());
				norms[i] += probs[i][j];
			}
			for (int j=0; j<labelings.length; j++)
				ret[i][j] = probs[i][j] / norms[i];
		}
		
/*		for (int j=0; j<labelings.length; j++) {
			for (int i=0; i<COMMITTEE_SIZE; i++)
				System.out.print(DF.format(ret[i][j])+" ");
			System.out.println("\t"+labelings[j]);
		}
*/		
		return ret;
	}
	
	/**
	 * getAvgProbs
	 */
	public double[] getAvgProbs(double[][] probs) {
		double[] ret = new double[probs[0].length];
		Arrays.fill(ret, 0.0);
		for (int i=0; i<COMMITTEE_SIZE; i++)
			for (int j=0; j<probs[i].length; j++)
				ret[j] += probs[i][j]/COMMITTEE_SIZE;
		return ret;
	}	

	/**
	 * toString
	 */
	public String toString() {
		return "QBC("+COMMITTEE_SIZE+","+((VOTE_ENTROPY) ? "ve" : "kl")+")";
	}
	
}
