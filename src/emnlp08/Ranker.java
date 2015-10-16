package emnlp08;

import edu.umass.cs.mallet.base.types.*;
import java.util.*;
import java.text.*;

/**
 * Wrapper class to help score and sort/organize objects.
 *
 * @author Burr Settles &lt;&gt;
 * @version $Rev$
 */
public class Ranker implements Comparable {
	private static DecimalFormat DF = new DecimalFormat("####.####");
	private double score;
	private int index;
	private Instance inst;
	
    public Ranker(int index, double score, Instance inst) {
		this.index = index;
        this.score = score;
		this.inst = inst;
    }

	/**
	* Comparison between two objects (for the comparable interface).
	*/
	public int compareTo(Object o) {
		Ranker r = (Ranker)o;
		if (score < r.score)
			return 1;
		else if (score > r.score)
			return -1;
		else
			return 0;
	}

	public String toString() {
		return index+" : "+inst.getName()+"\t// "+DF.format(score);
	}
	
	public double getScore() {
		return score;
	}
	
	public int getIndex() {
		return index;
	}
	public Instance getInstance() {
		return inst;
	}
	
}
