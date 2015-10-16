package emnlp08.pipes;

import edu.umass.cs.mallet.base.pipe.*;
import edu.umass.cs.mallet.base.types.*;
import java.util.regex.*;

/**
	<p>Input2TokenSequence is a text processing Pipe for the MALLET
	framework. It converts a tokenized sentence string into an input
	training/tagging sequence for the conditional random fields. Input
	one sentence per line. for training, '|' separates word|tag:

<pre>
	IL-2|B-DNA gene|I-DNA expression|O and|O NF-kappa|B-PROTEIN B|I-PROTEIN activation|O ...
	</pre>

	<p>For tagging new sequences, word tokens only will suffice:

<pre>
	IL-2 gene expression and NF-kappa B activation ...
	</pre>

	@author Burr Settles <a href="http://www.cs.wisc.edu/~bsettles">bsettles&#64;&#99;s&#46;&#119;i&#115;&#99;&#46;&#101;d&#117;</a> 
@version 1.5 (March 2005)
	*/

	public class Vanilla2TokenSequence extends Pipe {

	public Vanilla2TokenSequence () {
		super (null, LabelAlphabet.class);
	}

	public Instance pipe (Instance carrier) {
		String sentenceLines = (String) carrier.getData();
		String[] tokens = sentenceLines.trim().split ("[\t ]+");
		TokenSequence data = new TokenSequence (tokens.length);
		LabelSequence target = new LabelSequence ((LabelAlphabet)getTargetAlphabet(), tokens.length);

		String prevLabel = "NOLABEL";
		String word, pos, chunk, label, wc, bwc;
		String[] features;
		for (int i = 0; i < tokens.length; i++) {
			if (tokens[i].length() > 0) {
				Token token = new Token (tokens[i]);
				features = tokens[i].split ("\\|");
				// process all the non-label features...
				for (int j=1; j<features.length; j++)
					token.setFeatureValue (features[j], 1);
				// Append
				data.add (token);
				target.add (features[0]);
			}

		}
		//System.out.println ("");
		carrier.setData(data);
		carrier.setTarget(target);
		carrier.setName(sentenceLines);
		return carrier;
	}
}
