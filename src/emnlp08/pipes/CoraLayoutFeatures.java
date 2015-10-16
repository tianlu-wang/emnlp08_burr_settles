package emnlp08.pipes;

import edu.umass.cs.mallet.base.types.*;
import edu.umass.cs.mallet.base.pipe.*;
import edu.umass.cs.mallet.base.util.*;
import java.util.*;
import java.io.*;

public class CoraLayoutFeatures extends Pipe {
	static private boolean wordsOnly = false;

	public Instance pipe (Instance carrier) {
		HashMap hash = new HashMap();
		TokenSequence data = (TokenSequence)carrier.getData();
		int pos=0;
		for (int i=0; i<data.size(); i++) {
			Token token = data.getToken(i);
			// add start/end/middle features
			if (pos==0) {
				token.setFeatureValue("STARTLINE", 1);
			}
			else if (token.getText().equals("+L+") || i==data.size()-1) {
				token.setFeatureValue("ENDLINE", 1);
				pos = 0;
			} else {
				token.setFeatureValue("INLINE", 1);				
			}
			pos++;
		}
		return carrier;
	}
}
