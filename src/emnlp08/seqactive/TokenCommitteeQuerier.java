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

public class TokenCommitteeQuerier extends SequenceQuerier {	
	private static DecimalFormat DF = new DecimalFormat("####.######");
	private static int COMMITTEE_SIZE = 3;
	private static int PER_BATCH_SIZE = 15;
	private static boolean VOTE_ENTROPY = false;//false;
	private static boolean HARD_VE = false;//false;
	private static boolean AVERAGE = false;
/*	private static CRF4[] committee = null;*/
	
	public TokenCommitteeQuerier() {
		this(COMMITTEE_SIZE, PER_BATCH_SIZE, VOTE_ENTROPY, AVERAGE);
	}
	public TokenCommitteeQuerier(boolean voteEntropy) {
		this(COMMITTEE_SIZE, PER_BATCH_SIZE, voteEntropy, AVERAGE);
	}
	public TokenCommitteeQuerier(boolean voteEntropy, boolean avg) {
		this(COMMITTEE_SIZE, PER_BATCH_SIZE, voteEntropy, avg);
	}
	public TokenCommitteeQuerier(int committeeSize, boolean voteEntropy) {
		this(committeeSize, PER_BATCH_SIZE, voteEntropy, AVERAGE);
	}
	public TokenCommitteeQuerier(int committeeSize) {
		this(committeeSize, PER_BATCH_SIZE, VOTE_ENTROPY, AVERAGE);
	}
	public TokenCommitteeQuerier(int committeeSize, int batchSize, boolean voteEntropy, boolean avg) {
		this(committeeSize, batchSize, voteEntropy, HARD_VE, avg);
	}
	public TokenCommitteeQuerier(int committeeSize, int batchSize, boolean voteEntropy, boolean hard, boolean avg) {
		COMMITTEE_SIZE = committeeSize;
		PER_BATCH_SIZE = batchSize;
		VOTE_ENTROPY = voteEntropy;
		HARD_VE = hard;
		AVERAGE = avg;
	}

	public TokenCommitteeQuerier(boolean voteEntropy, boolean hard, boolean avg) {
		this(COMMITTEE_SIZE, PER_BATCH_SIZE, voteEntropy, hard, avg);
	}

	
	public void setHardVE(boolean hve) {
		HARD_VE = hve;
	}
	public void setAverage(boolean avg) {
		AVERAGE = avg;
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
		for (int i=0; i<poolData.size(); i++) {
			double score = 0;
			Alphabet labels = committee[0].getOutputAlphabet();
			Instance inst = (Instance)poolData.get(i);
			Sequence input = (Sequence) inst.getData();
			// "hard" vote entropy
			if (VOTE_ENTROPY && HARD_VE)
				score = hardVoteEntropy(committee, input, labels);
			// distibution-based token QBC methods
			else {
			    // now compute "consensus" label distribution for each token + sum entropies
				double[][][] probs = getProbs(committee, input, (LabelAlphabet)poolData.getTargetAlphabet());//labels);
				for (int t=0; t<input.size(); t++) {
					double[] avgProbs = getAvgProbs(probs[t]);
					// "soft" vote entropy
					if (VOTE_ENTROPY)
						score += Misc.entropy(avgProbs);
					// mean KL divergence (default?)
					else {
						for (int j=0; j<COMMITTEE_SIZE; j++)
							score += Misc.divergenceKL(probs[t][j], avgProbs)/COMMITTEE_SIZE;
					}
				}
			}
			// normalize for sequence length?
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
	 * hardVoteEntropy
	 */
	public static double hardVoteEntropy(CRF4[] committee, Sequence input, Alphabet labels) {
		double score = 0;
/*		Alphabet labels = committee[0].getOutputAlphabet();*/
		Sequence[] viterbis = new Sequence[COMMITTEE_SIZE];
		// find all viterbi paths
		for (int j=0; j<COMMITTEE_SIZE; j++)
			viterbis[j] = committee[j].viterbiPath(input).output();
		// tally up vote entropy
		for (int t=0; t<input.size(); t++) {
			double[] probs = new double[labels.size()];
			Arrays.fill(probs, 0.0);
			for (int j=0; j<COMMITTEE_SIZE; j++) {
				String l = (String)viterbis[j].get(t);
				probs[labels.lookupIndex(l)] += 1.0/COMMITTEE_SIZE;
			}
			double tokent = Misc.entropy(probs);
//			System.out.println(Misc.dumpArr(probs)+"\t// "+tokent);
			score += tokent;
		}
		return score;
	}

	/**
	 * getProbs
	 */
	public static double[][][] getProbs(CRF4[] committee, Sequence input, LabelAlphabet labels) {
		double[][][] ret = new double[input.size()][COMMITTEE_SIZE][labels.size()];
		// first construct all the lattices
		Transducer.Lattice[] lattices = new Transducer.Lattice[COMMITTEE_SIZE];
		for (int i=0; i<COMMITTEE_SIZE; i++)
			lattices[i] = committee[i].forwardBackward (input, null, false, labels);//(LabelAlphabet)poolData.getTargetAlphabet());
		// compute marginals for each token / committee member
		for (int t=0; t<input.size(); t++) {
			for (int c=0; c<COMMITTEE_SIZE; c++) {
				LabelVector lv = lattices[c].getLabelingAtPosition(t);
				for (int i=0; i<lv.numLocations(); i++)
					ret[t][c][i] = lv.value(i);
			}
		}
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
		String type = "kl";
		if (VOTE_ENTROPY)
			type = (HARD_VE) ? "hve" : "ve";
		if (AVERAGE)
			type = "avg-"+type;
		return "TOKEN-QBC("+COMMITTEE_SIZE+","+type+")";
	}
	
}
