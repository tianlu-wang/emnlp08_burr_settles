package emnlp08;

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

public class Misc {
    public static DecimalFormat DF = new DecimalFormat("####.######");
    public static Metric dpMetric = new NormalizedDotProductMetric();
    public static Metric euMetric = new Minkowski(2);
    
    /**
     * cosSimilarity
     */
    static public double cosSimilarity(SparseVector sv1, SparseVector sv2) {
        double sim = 1-dpMetric.distance(sv1, sv2);
        if (Double.isNaN(sim))
            return 0.0;
/*        System.out.println(DF.format(sim)+"\n\t--> "+
            sv1.toString(true)+"\n\t--> "+
            sv2.toString(true)+"\n");*/
        return sim;
    }
    
    /**
     * gaussSimilarity
     */
    public static double klSimilarity(SparseVector sv1, SparseVector sv2, int numFeats) {
        double BETA = 3.0;
        double LAMBDA = 0.5;
        int[] idx1 = sv1.getIndices();
        int[] idx2 = sv2.getIndices();
        double dist = 0;
        // cycle through sparse indices for i1
        for (int ii=0; ii<idx1.length; ii++) {
            int k = idx1[ii];
            double pWd1 = sv1.value(k) / sv1.absNorm();
            double pWd2 = (LAMBDA * sv2.value(k) / sv2.absNorm()) + ((1-LAMBDA)/numFeats);
/*            double pWd1 = 1.0/idx1.length;*/
/*            double pWd2 = (LAMBDA/idx2.length) + ((1-LAMBDA)/numFeats);*/
            if (pWd1 > 0)
                dist += pWd1 * Math.log(pWd1 / pWd2);
        }
        double sim = Math.exp(-BETA*dist);
/*        System.out.println(DF.format(dist)+" : "+DF.format(sim)+"\n\t--> "+
            sv1.toString(true)+"\n\t--> "+
            sv2.toString(true)+"\n");*/
        return sim;
    }


    /**
     * gaussSimilarity
     */
    public static double gaussSimilarity(SparseVector sv1, SparseVector sv2, int numFeats) {
        double ALPHA = 10; //0.01;
        int[] idx1 = sv1.getIndices();
        int[] idx2 = sv2.getIndices();
        boolean[] used = new boolean[numFeats];
        Arrays.fill(used, false);
        double dist = 0;
        // cycle through sparse indices for i1
        for (int ii=0; ii<idx1.length; ii++) {
            int k = idx1[ii];
            dist += Math.pow((sv1.value(k) - sv2.value(k)) / ALPHA, 2);
            used[k] = true;
        }
        // and again for i1
        for (int ii=0; ii<idx2.length; ii++) {
            int k = idx2[ii];
            // only params not previously looked at
            if (!used[k])
                dist += Math.pow((sv1.value(k) - sv2.value(k)) / ALPHA, 2);
        }
        double sim = Math.exp(-dist/2);
/*        System.out.println(DF.format(dist)+" : "+DF.format(sim)+"\n\t--> "+
            sv1.toString(true)+"\n\t--> "+
            sv2.toString(true)+"\n");*/
        return sim;
    }
    
    /**
     * tanhSimilarity
     */
    public static double tanhSimilarity(SparseVector sv1, SparseVector sv2, int numFeats) {
        double ALPHA = 0.5;
        double BETA = 0.5; //Math.sqrt(numFeats * 0.1 * 0.1);
        int[] idx1 = sv1.getIndices();
        int[] idx2 = sv2.getIndices();
        boolean[] used = new boolean[numFeats];
        Arrays.fill(used, false);
        double dist = 0;
        // cycle through sparse indices for i1
        for (int ii=0; ii<idx1.length; ii++) {
            int k = idx1[ii];
            dist += Math.abs(sv1.value(k) - sv2.value(k)) - BETA;
            used[k] = true;
        }
        // and again for i1
        for (int ii=0; ii<idx2.length; ii++) {
            int k = idx2[ii];
            // only params not previously looked at
            if (!used[k])
                dist += Math.abs(sv1.value(k) - sv2.value(k)) - BETA;
        }
        dist = Math.sqrt(dist);
        if (dist == 0)
            return 1.0;
        double sim = 1-(Math.tanh(ALPHA*(dist))+1)/2;
/*        System.out.println(DF.format(dist)+" ("+DF.format(BETA)+") : "+DF.format(sim)+"\n\t--> "+
            sv1.toString(true)+"\n\t--> "+
            sv2.toString(true)+"\n");*/
        return sim;
    }
    
    /**
     * mean value of array "x"
     */
    static public double average(double[] x) {
        double r=0;
        for (int i=0; i<x.length; i++)
            r+=x[i];
        return (r/x.length);
    }
    
    /**
     * standard deviation of array "x"
     */
    static public double stdev(double[] x) {
        double mean = average(x);
        double r=0;
        for (int i=0; i<x.length; i++)
            r += Math.pow(x[i]-mean,2);
        return Math.sqrt(r/(x.length-1));
    }

    /**
     * confidence interval for array "x"
     */
    static public double confidence(double[] x, double coeff) {
        return coeff*(stdev(x)/Math.sqrt(x.length));
    }   

    /**
     * default 95% confidence interval
     */
    static public double confidence(double[] x) {
        return confidence(x, 1.96);
    }

    static public String dumpArr(double[] x) {
        StringBuffer sb = new StringBuffer();
        for (int i=0; i<x.length; i++)
            sb.append("["+DF.format(x[i])+"]");
        return sb.toString();
    }
    
    /**
     * divergenceKL
     */
    static public double divergenceKL(double[] p, double[] q) {
        assert(p.length == q.length);
        double ret = 0;
        for (int i=0; i<p.length; i++)
            ret += p[i] * Math.log(p[i]/q[i]);
        return ret;
    }
    
    /**
     * entropy
     */
    static public double entropy(double[] p) {
        double ret = 0;
        for (int i=0; i<p.length; i++)
            ret -= (p[i] > 0)
                ? p[i] * Math.log(p[i]) 
                : 0;
        return ret;
    }
    static public double entropy(LabelVector lv) {
        double ret=0;
        // for each 
        for (int i=0; i<lv.numLocations(); i++) {
            double p = lv.value(i);
            ret -= (p > 0) ?
                p * (Math.log(p)) ///Math.log(2));
                : 0;
        }
        return ret;
    }
    
    

    /**
     * sequenceEntropy
     */
    static public double sequenceEntropy(CRF4 model, Instance inst) {
        Sequence input = (Sequence) inst.getData();
        Transducer.Lattice lattice = model.forwardBackward( input );
        EntropyLattice el = new EntropyLattice (input, model, lattice, false, 1.0);
        return el.getEntropy();
    }
    
    /**
     * kBestSequenceEntropy
     */
    public static double kBestSequenceEntropy(CRF4 model, Instance inst, int k) {
        Sequence input = (Sequence) inst.getData();
        Transducer.Lattice lattice = model.forwardBackward( input );
        // (1) calculate the "lattice cost":
        double latticeCost = lattice.getCost();
        // (2) calculate the costs for the top ENT_WINDOW sequences:
        Transducer.ViterbiPath_NBest vpnb = model.viterbiPath_NBest ( input, k );
        double[] costs = vpnb.costNBest();
        double[] probs = new double[costs.length];
        // (3) normalize+exponentiate for each, get margin
        double totalProb = 0;
        for (int j=0; j<costs.length; j++) {
            probs[j] = Math.exp( latticeCost - costs[j] );
            totalProb += probs[j];
        }
        double entropy = 0;
        for (int j=0; j<probs.length; j++) {
            double p = probs[j] / totalProb;
            if (p > 0)
                entropy -= p * (Math.log(p));///Math.log(2));
        }
        return entropy;
    }

    /**
     * makeKernelVectors
     */
    public static void makeKernelVectors(InstanceList ilist, FeatureSelection fs) {
        // do this for each instance
        for (int i=0; i<ilist.size(); i++) {
            Instance inst = ilist.getInstance(i);
            HashMap hash = new HashMap();
            FeatureVectorSequence fvs = (FeatureVectorSequence)inst.getData();
            // walk down each token
            for (int ii = 0; ii < fvs.size(); ii++) {
                FeatureVector fv = (FeatureVector)fvs.get(ii);
                int[] indices = fv.getIndices();
                // for each feature THAT'S IN THE FEATURE SELECTION (if not null)
                for (int j=0; j<indices.length; j++) {
                    if (fs == null || fs.contains(j)) {
                        int idx = indices[j];
                        String sidx = ""+idx;
                        double val = fv.value(idx);
                        if (hash.containsKey(sidx)) {
                            hash.put(sidx, new Double(val + ((Double)hash.get(sidx)).doubleValue()));
                        } else {
                            hash.put(sidx, new Double(val));
                        }                       
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
            // shove sparse vector into source field of the instance!
            SparseVector sv = new SparseVector(indices, values);
            inst.unLock();
            inst.setSource(sv);
            inst.setLock();         
        }
    }
    
    /**
     * makeAverageVector
     */
    public static SparseVector makeAverageVector(InstanceList ilist) {
        HashMap hash = new HashMap();
        // cycle through instances
        for (int i=0; i<ilist.size(); i++) {
            Instance inst = ilist.getInstance(i);
            SparseVector fv = (SparseVector)inst.getSource();
            int[] indices = fv.getIndices();
            for (int j=0; j<indices.length; j++) {
                int idx = indices[j];
                String sidx = ""+idx;
                double val = fv.value(idx);
                if (hash.containsKey(new Integer(indices[j]))) {
                    hash.put(sidx, new Double((val/ilist.size()) + ((Double)hash.get(sidx)).doubleValue()));
                } else {
                    hash.put(sidx, new Double(val/ilist.size()));
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
        // return SV representation
        return new SparseVector(indices, values);
    }

    /**
     * tfidf
     */
    public static void tfidf(InstanceList ilist) {
        double[] dfs = new double[ilist.getDataAlphabet().size()];
        // first go through and compute all the "document frequencies"
        for (int i=0; i<ilist.size(); i++) {
            Instance inst = (Instance)ilist.get(i);
            int[] indices = ((SparseVector)inst.getSource()).getIndices();
            for (int ii=0; ii<indices.length; ii++)
                dfs[indices[ii]]++;
        }
        // now adjust "term frequencies"
        for (int i=0; i<ilist.size(); i++) {
            Instance inst = (Instance)ilist.get(i);
            SparseVector sv = (SparseVector)inst.getSource();
            int[] indices = sv.getIndices();
            for (int ii=0; ii<indices.length; ii++) {
                int k = indices[ii];
                sv.setValue(k, (sv.value(k) / dfs[k]));
            }
//          System.out.println("#### "+inst.getName());
//          System.out.println(sv.toString(true));
        }
    }
    
}
