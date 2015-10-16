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

public class EGLQuerier extends SequenceQuerier {  
    private static DecimalFormat DF = new DecimalFormat("####.######");
    private static int ENT_WINDOW = 20;

    public EGLQuerier() {}
    public EGLQuerier(int k) {
        ENT_WINDOW = k;
    }

    public int[] select(CRF4 model, InstanceList poolData, InstanceList trainData, int num) {
        int[] ret = new int[num];
        Ranker[] scoring = new Ranker[poolData.size()];
        for (int i=0; i<poolData.size(); i++) {
            Instance inst = (Instance)poolData.get(i);
            double score = expectedGradientLength(model, inst, model.numParameters(trainData), poolData.getTargetAlphabet());
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
     * expectedGradientLength
     */
    public static double expectedGradientLength(CRF4 model, Instance inst, int numParams, Alphabet targetAlphabet) {
        double ret = 0.0; // this is the return value
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
                tmpInst.setTarget(makeLabelSequence(labels[j], targetAlphabet));
                // fabricate list w/ fake instance
                InstanceList tmpList = new InstanceList();
                tmpList.add(tmpInst);
                // get the length of this 
                SparseVector tmpSV = model.getGradient(tmpList);
                ret += p * tmpSV.twoNorm();
            }
        }
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
        return "EGL";
    }
    
}
