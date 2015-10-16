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

	public class NER2TokenSequence extends Pipe
{
	boolean saveSource = true;
	boolean doDigitCollapses = false;
	boolean doDowncasing = true;
	boolean doWordClass = false;
	boolean doBriefWordClass = false;

	public NER2TokenSequence (boolean cls)
	{
		super (null, LabelAlphabet.class);
		doWordClass = cls;
		doBriefWordClass = cls;
	}

	public NER2TokenSequence ()
	{
		super (null, LabelAlphabet.class);
	}

	public Instance pipe (Instance carrier)
	{
		String sentenceLines = (String) carrier.getData();
		String[] tokens = sentenceLines.trim().split ("[\t ]+");
		TokenSequence data = new TokenSequence (tokens.length);
		LabelSequence target = new LabelSequence ((LabelAlphabet)getTargetAlphabet(), tokens.length);
		StringBuffer source = saveSource ? new StringBuffer() : null;

		String prevLabel = "NOLABEL";
		String word, pos, chunk, label, wc, bwc;
		String[] features;
		for (int i = 0; i < tokens.length; i++) {
			if (tokens[i].length() > 0) {
				features = tokens[i].split ("\\|");
				if (features.length > 4)
					throw new IllegalStateException ("Line \""+tokens[i]+"\" is formatted badly!");
				word = features[0];
				wc = word;
				bwc = word;
				label = features[features.length-1].toUpperCase();
				pos = (features.length > 2) ? features[1] : "";
				chunk = (features.length > 3) ? features[2] : "";
			} else {
				word = "";
				pos = "";
				chunk = "";
				wc = "";
				bwc = "";
				label = "";
			}

			// Transformations
			if (doDigitCollapses) {
				if (word.matches ("19\\d\\d"))
					word = "<YEAR>";
				else if (word.matches ("19\\d\\ds"))
					word = "<YEARDECADE>";
				else if (word.matches ("19\\d\\d-\\d+"))
					word = "<YEARSPAN>";
				else if (word.matches ("\\d+\\\\/\\d"))
					word = "<FRACTION>";
				else if (word.matches ("\\d[\\d,\\.]*"))
					word = "<DIGITS>";
				else if (word.matches ("19\\d\\d-\\d\\d-\\d--d"))
					word = "<DATELINEDATE>";
				else if (word.matches ("19\\d\\d-\\d\\d-\\d\\d"))
					word = "<DATELINEDATE>";
				else if (word.matches (".*-led"))
					word = "<LED>";
				else if (word.matches (".*-sponsored"))
					word = "<LED>";
			}

			// do the word class business
			if (doWordClass) {
				wc = wc.replaceAll("[A-Z]", "A");
				wc = wc.replaceAll("[a-z]", "a");
				wc = wc.replaceAll("[0-9]", "0");
				wc = wc.replaceAll("[^A-Za-z0-9]", "x");
			}
			if (doBriefWordClass) {
				bwc = bwc.replaceAll("[A-Z]+", "A");
				bwc = bwc.replaceAll("[a-z]+", "a");
				bwc = bwc.replaceAll("[0-9]+", "0");
				bwc = bwc.replaceAll("[^A-Za-z0-9]+", "x");
			}

			Token token = new Token (word);
			if (doDowncasing)
				word = word.toLowerCase();
			token.setFeatureValue ("W="+word, 1);

			if (!pos.equals(""))
				token.setFeatureValue ("POS="+pos, 1);
			if (!chunk.equals(""))
				token.setFeatureValue ("CHUNK="+chunk, 1);
			if (doWordClass)
				token.setFeatureValue ("WC="+wc, 1);
			if (doBriefWordClass)
				token.setFeatureValue ("BWC="+bwc, 1);

			// Append
			data.add (token);
			target.add (label);
			if (saveSource) {
				source.append (token.getText()); source.append (" ");
			}

		}
		//System.out.println ("");
		carrier.setData(data);
		carrier.setTarget(target);
		if (saveSource)
			carrier.setName(source);
		return carrier;
	}
}
