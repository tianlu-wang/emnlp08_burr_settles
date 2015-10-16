package emnlp08.pipes;

import edu.umass.cs.mallet.base.types.*;
import edu.umass.cs.mallet.base.pipe.*;
import edu.umass.cs.mallet.base.util.*;
import java.util.*;
import java.io.*;

public class TerminalLines extends Pipe {
	static private boolean wordsOnly = false;

	public Instance pipe (Instance carrier) {
		HashMap hash = new HashMap();
		TokenSequence data = (TokenSequence)carrier.getData();
		for (int i=data.size()-3; i<data.size(); i++) {
			Token token = data.getToken(i);
			token.setFeatureValue("FROMEND="+(data.size()-i), 1);
		}
		return carrier;
	}
}
