/*package edu.umass.cs.mallet.users.gmann.base.fst;*/
package emnlp08;

import edu.umass.cs.mallet.base.fst.*;
import edu.umass.cs.mallet.base.types.InstanceList;
import edu.umass.cs.mallet.base.types.Instance;
import edu.umass.cs.mallet.base.types.Sequence;
import edu.umass.cs.mallet.base.types.ArraySequence;
import edu.umass.cs.mallet.base.types.SequencePair;
import edu.umass.cs.mallet.base.types.SequencePairAlignment;
import edu.umass.cs.mallet.base.types.Label;
import edu.umass.cs.mallet.base.types.LabelAlphabet;
import edu.umass.cs.mallet.base.types.LabelVector;
import edu.umass.cs.mallet.base.types.DenseVector;
import edu.umass.cs.mallet.base.types.Alphabet;
import edu.umass.cs.mallet.base.types.MatrixOps;
import edu.umass.cs.mallet.base.util.MalletLogger;
import edu.umass.cs.mallet.base.util.search.*;

public class EntropyLattice {
	public double             entropyBackward;
	public double             entropyForward;
	public double             entropyWainwright;

	// "ip" == "input position", "op" == "output position", "i" == "state index"
	Transducer         transducer;
	Transducer.Lattice lattice;

	int                latticeLength;
	int                inputLength;
	LatticeNode[][]    nodes;			 // indexed by ip,i
	int                numStates;
	Sequence           input;

	public double getEntropy(){
		return entropyForward;
	}

	// fast entropy calculation
	public static double calcEntropy(Transducer transducer, Transducer.Lattice lattice){
		int numStates   = transducer.numStates();
		int inputLength = lattice.length() - 1;
		
		double entropy = 0;
		double sumNegLog = 0; 

		double[][][] xis  = lattice.getXis();
		double[][] gammas = lattice.getGammas();

		// \sum_t=1..T,y1,y2 p(y1,y2) log (1/p(y1,y2))
		for (int ip = 1; ip < inputLength; ip++){
			for (int y1 = 0; y1 < numStates; y1++){
				for (int y2 = 0; y2 < numStates; y2++){
					double xi = xis[ip][y1][y2];
					//System.err.println("ip y1 y2:"+ip+" "+y1+" "+y2+" xi "+Math.exp(-xi));
					// if this transition has non-zero probability at this position
					if (xi < Transducer.INFINITE_COST){
						double pi = Math.exp(-xi);
						entropy += pi * xi;
					}
				}
			}
		}

		// - \sum_t=2..T-1,y1 p(y1) log (1/p(py1))
		for (int ip = 2; ip < inputLength; ip++){
			for (int y1 = 0; y1 < numStates; y1++){
				double gamma = gammas[ip][y1];
				//System.err.println("ip y1:"+ip+" "+y1+" gamma "+Math.exp(-gamma));
				// if this label has non-zero probability
				if (gamma < Transducer.INFINITE_COST){
					double pi = Math.exp(-gamma);
					entropy -= pi * gamma;
				}
			}
		}

		return entropy;
	}

	// dynamic program entropy calculation (for entropy gradient)
	public EntropyLattice (Sequence input, Transducer transducer, Transducer.Lattice lattice, boolean update, double scalingFactor)	{
		this.transducer   = transducer;
		this.lattice      = lattice;
		this.input        = input;

		latticeLength = input.size() + 1;
		inputLength = input.size();
		numStates = transducer.numStates();
		nodes = new LatticeNode[latticeLength][numStates];

		entropyForward  = forwardLattice(lattice.getGammas(), lattice.getXis() );
		entropyBackward = backwardLattice(lattice.getGammas(), lattice.getXis() );
		entropyWainwright = calcEntropy(transducer, lattice);
		assert(!Double.isNaN(entropyForward)) :
		"entropy(forward)="+ entropyForward 
			+ " entropy(backward)="+entropyBackward
			+ " entropy(wainwright)="+entropyWainwright;
		assert(!Double.isNaN(entropyBackward));
		assert(!Double.isNaN(entropyWainwright));
		
//		verifyEntropyCalculations(lattice.getGammas(), lattice.getXis());

		if (update){
			updateCounts(lattice.getGammas(), lattice.getXis(), scalingFactor);
		}
	}

	private void verifyEntropyCalculations(double[][] gammas, double[][][] xis){
		for (int ip = 0; ip < inputLength; ip++){
			double positionEntropy = 0;
			for (int a = 0 ; a < numStates; a++){
				if (nodes[ip][a] == null) {
					continue; }
				
				Transducer.State s = transducer.getState(a);
				
				Transducer.TransitionIterator iter = s.transitionIterator(input, ip, null, ip);
				while (iter.hasNext()){
					Transducer.State destination = iter.nextState();
					int b = destination.getIndex();
					
					// nodes created for all possible states, including ones that are actually invalid
					// therefore must check for invalid transitiby examining xis
					if (xis[ip][a][b] == Transducer.INFINITE_COST){
						continue; }

					double xi = xis[ip][a][b];
					double xiProb = Math.exp(-xi);

					positionEntropy += xiProb * ( xi + nodes[ip][a].alpha + nodes[ip+1][b].beta);
				}
			}
			System.err.println("total entropy for position "+ip+" = "+positionEntropy);  
		}
		System.err.println("total sequence entropy="+entropyForward);
	}

	private void updateCounts(double[][] gammas, double[][][] xis, double scalingFactor){
		for (int ip = 0; ip < inputLength; ip++){
			for (int a = 0 ; a < numStates; a++){
				if (nodes[ip][a] == null) {
					continue; }
				
				Transducer.State s = transducer.getState(a);
				
				Transducer.TransitionIterator iter = s.transitionIterator(input, ip, null, ip);
				while (iter.hasNext()){
					Transducer.State destination = iter.nextState();
					int b = destination.getIndex();
					
					// nodes created for all possible states, including ones that are actually invalid
					// therefore must check for invalid transitions by examining xis
					if (xis[ip][a][b] == Transducer.INFINITE_COST){
						continue; }

					double xi = xis[ip][a][b];
					double xiProb = Math.exp(-xi);
					

					// \Sum_(y_i, y_{i+1}) fk(y_i,y_{i+1},x) p(y_i, y_{i+1}) * (log p(y_i, y_{i+1}) + H^a(Y_{1..(i-1)},y_i) + H^b(Y_{(i+2)..T}|y_{i+1}))
					//                                                       - (\Sum_Y p(Y) log p(Y) )
					double negativeEntropyContribution =   - xiProb * ( xi + nodes[ip][a].alpha + nodes[ip+1][b].beta - entropyForward);

					assert(!Double.isNaN(negativeEntropyContribution)) :
					"xi="+xi
						+"nodes["+ip+"]["+a+"].alpha="+nodes[ip][a].alpha
						+"nodes["+(ip+1)+"]["+b+"].beta="+nodes[ip+1][b].beta;

// 					System.err.println("ip="+ip+" a="+a+" b="+b);
// 					System.err.println("xis[ip][a][b]="+xis[ip][a][b]);
// 					System.err.println("xiProb="+xiProb);
// 					System.err.println("nodes[ip][a].beta="+nodes[ip][a].alpha);
// 					System.err.println("nodes[ip+1][b].beta="+nodes[ip+1][b].beta);
// 					System.err.println("entropyForward="+entropyForward);
// 					System.err.println("negativeEntropyContribution="+negativeEntropyContribution);
// 					System.err.println("scalingFactor="+scalingFactor);

					iter.incrementCount(negativeEntropyContribution * scalingFactor);
				}
			}
		}
	}

	private double forwardLattice(double[][] gammas, double[][][] xis){

		for (int a = 0; a < numStates; a++){
			getLatticeNode(0,a).alpha = 0; // forward entropy of start states is 0
		}

		for (int ip = 1; ip < latticeLength; ip++){
			for (int a = 0; a < numStates; a++) {
				LatticeNode node = getLatticeNode(ip,a);
				
				for (int b = 0; b < numStates; b++){

					double gamma   = gammas[ip][a];
					double xi      = xis[ip-1][b][a];

					if ((xi < Transducer.INFINITE_COST) && 
							(gamma < Transducer.INFINITE_COST)){

						double gammaPr = Math.exp(-gamma);
						double xiPr    = Math.exp(-xi);
						
						double condPr  = xiPr / gammaPr;
						double cond    = xi - gamma;
						
						node.alpha += condPr * ( cond + getLatticeNode(ip-1,b).alpha);
					}
				}
			}
		}

		double entropy = 0;
		for (int a = 0; a < numStates; a++){
			double gamma   = gammas[inputLength][a];
			double gammaPr = Math.exp(-gamma);

			if (gamma < Transducer.INFINITE_COST){
				entropy += gammaPr * gamma;
				entropy += gammaPr * getLatticeNode(inputLength, a).alpha;
			}
		}

		return entropy;
	}

	private double backwardLattice(double[][] gammas, double[][][] xis){
		// backward pass
		for (int a = 0; a < numStates; a++){
			getLatticeNode(inputLength,a).beta = 0; // backward entropy of end states is 0
		}

		for (int ip = inputLength ; ip >= 0; ip--){
			for (int a = 0; a < numStates; a++) {
				LatticeNode node = getLatticeNode(ip,a);
				
				for (int b = 0; b < numStates; b++){
					double gamma   = gammas[ip][a];
					double xi      = xis[ip][a][b];

					if ((xi < Transducer.INFINITE_COST) && 
							(gamma < Transducer.INFINITE_COST)){

						double gammaPr = Math.exp(-gamma);
						double xiPr    = Math.exp(-xi);
						
						double condPr  = xiPr / gammaPr;
						double cond    = xi - gamma;
						
						node.beta += condPr * ( cond + getLatticeNode(ip+1,b).beta);
						
						//System.err.println("ip="+ip+" a="+a+" b="+b);
						//System.err.println("gammaPr="+gammaPr+" xiPr="+xiPr);
						//System.err.println("node.alpha="+node.alpha);
					}
				}

				//System.err.println("Ha_"+ip+"("+a+")="+node.alpha);
			}
		}

		double entropy = 0;
		for (int a = 0; a < numStates; a++){
			double gamma   = gammas[0][a];
			double gammaPr = Math.exp(-gamma);

			if (gamma < Transducer.INFINITE_COST){
				entropy += gammaPr * gamma;
				entropy += gammaPr * getLatticeNode(0, a).beta;
			}
		}

		return entropy;
	}

	public double getAlpha (int ip, Transducer.State s) {
		LatticeNode node = getLatticeNode (ip, s.getIndex ());
		return node.alpha;
	}

	public double getBeta (int ip, Transducer.State s) {
		LatticeNode node = getLatticeNode (ip, s.getIndex ());
		return node.beta;
	}

	private LatticeNode getLatticeNode (int ip, int stateIndex)
	{
		if (nodes[ip][stateIndex] == null)
			nodes[ip][stateIndex] = new LatticeNode (ip, transducer.getState (stateIndex));
		return nodes[ip][stateIndex];
	}

	// A container for some information about a particular input position and state
	private class LatticeNode
	{
		int inputPosition;
		Transducer.State state;
		double alpha = 0;
		double beta = 0;
		LatticeNode (int inputPosition, Transducer.State state)	{
			this.inputPosition = inputPosition;
			this.state = state;
		}
	}
}
