package emnlp08;

import emnlp08.seqactive.*;
import emnlp08.pipes.*;
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

public class SequenceExp {
	public static DecimalFormat DF = new DecimalFormat("####.######");
	private static String LEXDIR = "data/lexicons/";
	public static String QUERY, VOCAB;
	public static int INIT, BATCH, NUMQ, FOLDS;
	public static int SUBSAMPLE_SIZE = 250, COMMITTEE_SIZE = 5;
	public static Random rand = new Random(27);
	public static HashMap densityHash = new HashMap();
	public static int ENT_WINDOW = 15;

	private static String CAPS = "[A-ZçƒêîòËéíñô‚„ì†]";
	private static String LOW = "[a-zˆ“˜‡’—œ–•Ÿ]";
	private static String CAPSNUM = "[A-ZçƒêîòËéíñô‚„ì†0-9]";
	private static String ALPHA = "[A-ZçƒêîòËéíñô‚„ì†a-zˆ“˜‡’—œ–•Ÿ]";
	private static String ALPHANUM = "[A-ZçƒêîòËéíñô‚„ì†a-zˆ“˜‡’—œ–•Ÿ0-9]";
	private static String PUNCTUATION = "[,\\.;:?!()]";
	private static String QUOTE = "[\"`']";
	private static String URLISH = "[A-Za-z0-9-_\\.]";
	private static String GREEK = "(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)";


	private static void printUsage() {
		System.err.println("Perform text classification.\n\nUsage: java dist.SequenceExp [OPTIONS] c1dir c2dir [cXdir...]");
		System.out.println("  -q, --query     query selection method to use (default='rand')");
		System.out.println("  -i, --init      # initial queries before active learning (default=5)");
		System.out.println("  -b, --batch     # queries to select each batch (default=5)");
		System.out.println("  -n, --numq      # rounds of active learning to perform (default=50)");
		System.out.println("  -f, --folds     # cross-validation folds to use (default=10)");
		System.out.println("  -v, --vocab     feature vocab to use (default='bioner' || 'parse', 'ner')");
	}

	public static void main (String[] args) throws FileNotFoundException, Exception {

		// prepare all the command line bidness...
		CmdLineParser parser = new CmdLineParser();
		CmdLineParser.Option _QUERY = parser.addStringOption('q', "query");
		CmdLineParser.Option _INIT = parser.addIntegerOption('i', "init");
		CmdLineParser.Option _BATCH = parser.addIntegerOption('b', "batch");
		CmdLineParser.Option _NUMQ = parser.addIntegerOption('n', "numq");
		CmdLineParser.Option _FOLDS = parser.addIntegerOption('f', "folds");
		CmdLineParser.Option _VOCAB = parser.addStringOption('v', "vocab");
		try {
			parser.parse(args);
		}
		catch ( CmdLineParser.OptionException e ) {
			System.err.println("ERR: "+e.getMessage()+"\n");
			printUsage();
			System.exit(2);
		}
		QUERY = ((String)parser.getOptionValue(_QUERY, "rand"));
		INIT = ((Integer)parser.getOptionValue(_INIT, new Integer(5))).intValue();
		BATCH = ((Integer)parser.getOptionValue(_BATCH, new Integer(5))).intValue();
		NUMQ = ((Integer)parser.getOptionValue(_NUMQ, new Integer(30))).intValue();
		FOLDS = ((Integer)parser.getOptionValue(_FOLDS, new Integer(5))).intValue();
		VOCAB = ((String)parser.getOptionValue(_VOCAB, "bioner"));
		String[] otherArgs = parser.getRemainingArgs();
		if (otherArgs.length < 1) {
			System.err.println("ERR: You must provide at least one input file!!\n");
			printUsage();
			System.exit(2);
		}

		// prepare the pipe used to transform data
		Pipe myPipe = getVocabPipe();
		MultiSegmentationEvaluator evaluator = getEvaluator();

		// read in all the data files...
		System.out.println("Loading data files...");
		InstanceList ilist = new InstanceList (myPipe);
		for (int i=0; i<otherArgs.length; i++)
			ilist.add (new LineGroupIterator (new FileReader (new File (otherArgs[i])), Pattern.compile("^.*$"), false));
		//      tfidf(ilist);
		InstanceList.CrossValidationIterator iter = ilist.crossValidationIterator(FOLDS, 27);

		//      System.out.println("Features: "+ilist.getDataAlphabet().size());
		//      System.exit(1);

		// the evaluation array
		double[][] evals = new double[NUMQ+1][FOLDS];

		// cycle through all the folds!!!
		int fold = 0;
		while (iter.hasNext()) {
			InstanceList[] split = iter.nextSplit();
			// prepare the data partitions
			InstanceList poolData = split[0];
			InstanceList evalData = split[1];
			// set up which sort of instance queryier we're using...
			SequenceQuerier querier = new RandomQuerier();

			// other baseline
			if (QUERY.equals("longest"))
				querier = new LongestQuerier();
			// basic uncertainty methods
			if (QUERY.equals("leastconf"))
				querier = new LeastConfQuerier();
			if (QUERY.equals("margin"))
				querier = new EntropyQuerier(2);
			// entropy-based methods
			if (QUERY.equals("ent"))
				querier = new EntropyQuerier();
			if (QUERY.equals("ent-kbest"))
				querier = new EntropyQuerier(ENT_WINDOW);
			if (QUERY.equals("ent-tok"))
				querier = new TokenEntropyQuerier();
			if (QUERY.equals("ent-avgtok"))
				querier = new TokenEntropyQuerier(true);
			// query-by committee methods
			if (QUERY.equals("qbc-kl"))
				querier = new CommitteeQuerier();
			if (QUERY.equals("qbc-ve"))
				querier = new CommitteeQuerier(true);
			// expected gradient length
			if (QUERY.equals("egl"))
				querier = new EGLQuerier();
			// token-level QBCs
			if (QUERY.equals("qbc-tok-kl"))
				querier = new TokenCommitteeQuerier(false, false, false);
			if (QUERY.equals("qbc-tok-ve"))
				querier = new TokenCommitteeQuerier(true, false, false);
			if (QUERY.equals("qbc-tok-hve"))
				querier = new TokenCommitteeQuerier(true, true, false);
			if (QUERY.equals("qbc-avgtok-kl"))
				querier = new TokenCommitteeQuerier(false, false, true);
			if (QUERY.equals("qbc-avgtok-ve"))
				querier = new TokenCommitteeQuerier(true, false, true);
			if (QUERY.equals("qbc-avgtok-hve"))
				querier = new TokenCommitteeQuerier(true, true, true);
			// density-weighted methods
			else if (QUERY.startsWith("kde")) {
				long start = (new Date()).getTime();
				System.out.println("Pre-computing fold "+fold+" densities...");
				if (QUERY.endsWith("cos"))
					System.out.println("Using COSINE kernel");
				else if (QUERY.endsWith("kl"))
					System.out.println("Using KL-DIVERGENCE kernel");
				else
					System.out.println("Using GAUSSIAN kernel");
				computeDensities(poolData);
				long elapsed = (new Date()).getTime() - start;
				System.out.println("Time to compute densities: "+elapsed);
				querier = new InformationDensityQuerier(densityHash);
			}
			else if (QUERY.equals("avgcos"))
				querier = new AvgCosDensityQuerier();
			// fisher information method
			else if (QUERY.equals("fisher"))
				querier = new FisherMatrixQuerier(ENT_WINDOW);
			// initialize training set w/ INIT random instances
			InstanceList trainData = new InstanceList();
			for (int i=0; i<INIT; i++) {
				int r = rand.nextInt(poolData.size());
				Instance inst = poolData.remove(r);
				System.out.println("- "+inst.getName());
				trainData.add(inst);
			}

			// train this initial model
			CRF4 model = new CRF4(myPipe, null);
			model.addStatesForLabelsConnectedAsIn(poolData);
			// start out using all features
			//          model.trainWithFeatureInduction(trainData, null, null, null, 150, 50, 3, 100, 0.2, false, new double[]{1.0});
			model.train(trainData, null, null, null, 150);

			double overallF1 = evaluator.overallF1(model, evalData);
			System.out.println("|params|="+model.numParameters(trainData)+" |pool|="+poolData.size()+" |train|="+trainData.size()+" |eval|="+evalData.size());
			System.out.println("FOLD "+fold+" INITIAL overallF1="+DF.format(overallF1));
			evals[0][fold] = overallF1; // store this accuracy
			evaluator.evaluate (model, true, 0, true, 0.0, null, null, evalData, true);

			// now commence the active learning...
			for (int q=1; q<=NUMQ; q++) {
				// select the next batch of queries
				long start = (new Date()).getTime();                
				/*              int[] queries = selectQueries(model, poolData, trainData, 117*q);*/
				int[] queries = querier.select(model, poolData, trainData, BATCH);
				// must pull them out in descending order
				for (int i=queries.length-1; i>=0; i--)
					trainData.add(poolData.remove(queries[i]));

				model.setTrainable(true);
				//              model.trainWithFeatureInduction(trainData, null, null, null, 150, 50, 3, 50, 0.2, false, new double[]{1.0});
				model.train(trainData, null, null, null, 150);

				// report the time elapsed to query & train
				long elapsed = (new Date()).getTime() - start;
				System.out.println("time to query+train: "+elapsed);

				overallF1 = evaluator.overallF1(model, evalData);
				System.out.println("|params|="+model.numParameters(trainData)+" |pool|="+poolData.size()+" |train|="+trainData.size()+" |eval|="+evalData.size());
				System.out.println("FOLD "+fold+" QUERY "+q+" overallF1="+DF.format(overallF1));
				evaluator.evaluate (model, true, 0, true, 0.0, null, null, evalData, true);

				evals[q][fold] = overallF1; // store this accuracy
			}

			// print out the model (just to see)
			/*            model.print();*/

			// increment fold count
			fold++;
		}

		// report final average and conf. scores
		System.out.println("\nFINAL EVALUATION!!!\n\n");
		for (int q=0; q<evals.length; q++) {
			System.out.println((q*BATCH)+"\t"+Misc.average(evals[q])+"\t"+Misc.confidence(evals[q]));
		}
	}

	static public Pipe getVocabPipe() {
		Pipe pipe = null;
		try {
			// vanilla conll NER...
			if (VOCAB.equals("ner") || VOCAB.equals("iesig")) {
				pipe = new SerialPipes (new Pipe[] {
						new NER2TokenSequence (),
						new RegexMatches ("INITCAP", Pattern.compile (CAPS+".*")),
						new RegexMatches ("CAPITALIZED", Pattern.compile (CAPS+LOW+"*")),
						new RegexMatches ("ALLCAPS", Pattern.compile (CAPS+"+")),
						new RegexMatches ("MIXEDCAPS", Pattern.compile ("[A-Z][a-z]+[A-Z][A-Za-z]*")),
						new RegexMatches ("CONTAINSDIGITS", Pattern.compile (".*[0-9].*")),
						new RegexMatches ("ALLDIGITS", Pattern.compile ("[0-9]+")),
						new RegexMatches ("NUMERICAL", Pattern.compile ("[-0-9]+[\\.,]+[0-9\\.,]+")),
						new RegexMatches ("MULTIDOTS", Pattern.compile ("\\.\\.+")),
						new RegexMatches ("ENDSINDOT", Pattern.compile ("[^\\.]+.*\\.")),
						new RegexMatches ("CONTAINSDASH", Pattern.compile (ALPHANUM+"+-"+ALPHANUM+"*")),
						new RegexMatches ("ACRO", Pattern.compile ("[A-Z][A-Z\\.]*\\.[A-Z\\.]*")),
						new RegexMatches ("LONELYINITIAL", Pattern.compile (CAPS+"\\.")),
						new RegexMatches ("SINGLECHAR", Pattern.compile (ALPHA)),
						new RegexMatches ("CAPLETTER", Pattern.compile ("[A-Z]")),
						new RegexMatches ("PUNC", Pattern.compile (PUNCTUATION)),
						new RegexMatches ("QUOTE", Pattern.compile (QUOTE)),
						new RegexMatches ("EMAIL", Pattern.compile ("URLISH"+"+\\@"+URLISH+"\\."+URLISH+"+")),
						new TrieLexiconMembership ("TITLE", new File(LEXDIR + "honorifics.txt"), false),
						new TrieLexiconMembership ("FEMALE", new File(LEXDIR + "names_female.txt"), false),
						new TrieLexiconMembership ("MALE", new File(LEXDIR + "names_male.txt"), false),
						new TrieLexiconMembership ("SURNAME", new File(LEXDIR + "names_last.txt"), false),
						new TrieLexiconMembership ("UNIVERSITY", new File(LEXDIR + "universities.txt"), false),
						new TrieLexiconMembership ("COMPANY", new File(LEXDIR + "companies_public.txt"), true),
						new TrieLexiconMembership ("CITY", new File(LEXDIR + "cities.txt"), false),
						new TrieLexiconMembership ("COUNTRY", new File(LEXDIR + "countries.txt"), false),
						new TrieLexiconMembership ("STATE", new File(LEXDIR + "states_provinces.txt"), false),
						new TrieLexiconMembership ("DATE", new File(LEXDIR + "days_months.txt"), false),
						new TrieLexiconMembership ("LANGUAGE", new File(LEXDIR + "languages.txt"), false),
						new OffsetConjunctions (new int[][] {{-1}, {1}}),
						//                  new PrintTokenSequenceFeatures(), // for debugging
						//                  new TokenSequence2SparseVector(),
						new TokenSequence2FeatureVectorSequence (true, true),
						//                  new FVS2KernelVector(),
				}); 
			}
			// BIO NER VOCABULARY
			if (VOCAB.equals("bioner")) {
				pipe = new SerialPipes (new Pipe[] {
						new NER2TokenSequence (true),
						new RegexMatches ("INITCAPS", Pattern.compile ("[A-Z].*")),
						new RegexMatches ("ALLCAPS", Pattern.compile ("[A-Z]+")),
						new RegexMatches ("CAPSMIX", Pattern.compile (".*[A-Z]+[a-z]+.*")),
						new RegexMatches ("CAPSMIX", Pattern.compile (".*[a-z]+[A-Z]+.*")),
						new RegexMatches ("HASDIGIT", Pattern.compile (".*[0-9].*")),
						new RegexMatches ("SINGLEDIGIT", Pattern.compile ("[0-9]")),
						new RegexMatches ("DOUBLEDIGIT", Pattern.compile ("[0-9][0-9]")),
						new RegexMatches ("NATURALNUMBER", Pattern.compile ("[0-9]+")),
						new RegexMatches ("REALNUMBER", Pattern.compile ("[-0-9]+[.,]+[0-9.,]+")),
						new RegexMatches ("HASDASH", Pattern.compile (".*-.*")),
						new TokenTextCharPrefix ("PREFIX=", 3),
						new TokenTextCharPrefix ("PREFIX=", 4),
						new TokenTextCharSuffix ("SUFFIX=", 3),
						new TokenTextCharSuffix ("SUFFIX=", 4),
						new OffsetConjunctions (new int[][] {{-1}, {1}}),
						new RegexMatches ("ALPHANUMERIC", Pattern.compile (".*[A-Za-z].*[0-9].*")),
						new RegexMatches ("ALPHANUMERIC", Pattern.compile (".*[0-9].*[A-Za-z].*")),
						new RegexMatches ("HASROMAN", Pattern.compile (".*\\b[IVXDLCM]+\\b.*")),
						new RegexMatches ("HASGREEK", Pattern.compile (".*\\b"+GREEK+"\\b.*")),
						new RegexMatches ("PUNCTUATION", Pattern.compile ("[,.;:?!-+]")),
						//new PrintTokenSequenceFeatures(), // for debugging
						//                  new TokenSequence2SparseVector(),
						new TokenSequence2FeatureVectorSequence (true, true),
						//                  new FVS2KernelVector(),
				});
			}
			// shallow parsing data set
			if (VOCAB.equals("parse")) {
				pipe = new SerialPipes (new Pipe[] {
						new NER2TokenSequence (),
						new RegexMatches ("INITCAP", Pattern.compile (CAPS+".*")),
						new RegexMatches ("CAPITALIZED", Pattern.compile (CAPS+LOW+"*")),
						new RegexMatches ("ALLCAPS", Pattern.compile (CAPS+"+")),
						new RegexMatches ("MIXEDCASE", Pattern.compile (".*[a-z][A-Z].*")),
						new RegexMatches ("ALPHANUMERIC", Pattern.compile (".*[A-Za-z].*[0-9].*")),
						new RegexMatches ("ALPHANUMERIC", Pattern.compile (".*[0-9].*[A-Za-z].*")),
						new RegexMatches ("ALLDIGITS", Pattern.compile ("[0-9]+")),
						new RegexMatches ("NUMERICAL", Pattern.compile ("[-0-9]+[\\.,]+[0-9\\.,]+")),
						new RegexMatches ("MULTIDOTS", Pattern.compile ("\\.\\.+")),
						new RegexMatches ("ENDSINDOT", Pattern.compile ("[^\\.]+.*\\.")),
						new RegexMatches ("CONTAINSDASH", Pattern.compile (ALPHANUM+"+-"+ALPHANUM+"*")),
						new RegexMatches ("ACRO", Pattern.compile ("[A-Z][A-Z\\.]*\\.[A-Z\\.]*")),
						new RegexMatches ("LONELYINITIAL", Pattern.compile (CAPS+"\\.")),
						new RegexMatches ("SINGLECHAR", Pattern.compile (ALPHA)),
						new RegexMatches ("CAPLETTER", Pattern.compile ("[A-Z]")),
						new RegexMatches ("PUNC", Pattern.compile (PUNCTUATION)),
						new RegexMatches ("QUOTE", Pattern.compile (QUOTE)),
						new FeaturesInWindow ("@-1", -1, 1, Pattern.compile("W=.*"), true),
						new FeaturesInWindow ("@+1", 1, 2, Pattern.compile("W=.*"), true),
						new FeaturesInWindow ("@-1", -1, 1, Pattern.compile("T=.*"), true),
						new FeaturesInWindow ("@+1", 1, 2, Pattern.compile("T=.*"), true),
						new FeaturesInWindow ("@-2", -2, -1, Pattern.compile("W=.*"), true),
						new FeaturesInWindow ("@+2", 2, 3, Pattern.compile("W=.*"), true),
						new FeaturesInWindow ("@-2", -2, -1, Pattern.compile("T=.*"), true),
						new FeaturesInWindow ("@+2", 2, 3, Pattern.compile("T=.*"), true),
						//                  new PrintTokenSequenceFeatures(),
						new TokenSequence2FeatureVectorSequence (true, true),
						//                  new FVS2KernelVector(),
				});
			}
			// CORA research paper corpus...
			if (VOCAB.startsWith("cora")) {
				pipe = new SerialPipes (new Pipe[] {
						new NER2TokenSequence (),
						new RegexMatches ("INITCAP", Pattern.compile (CAPS+".*")),
						new RegexMatches ("ALLCAPS", Pattern.compile (CAPS+"+")),
						new RegexMatches ("CONTAINSDIGIT", Pattern.compile (".*[0-9].*")),
						new RegexMatches ("ALLDIGITS", Pattern.compile ("[0-9]+")),
						new RegexMatches ("PHONEORZIP", Pattern.compile ("[0-9][0-9][0-9][0-9][0-9]")),
						new RegexMatches ("PHONEORZIP", Pattern.compile (".*[0-9][0-9][0-9].+[0-9][0-9][0-9][0-9].*")),
						new RegexMatches ("CONTAINSDOT", Pattern.compile (".*\\..*")),
						new RegexMatches ("ACRO", Pattern.compile ("[A-Z][A-Z\\.]*\\.[A-Z\\.]*")),
						new RegexMatches ("LONELYINITIAL", Pattern.compile (CAPS+"\\.")),
						new RegexMatches ("SINGLECHAR", Pattern.compile (ALPHA)),
						new RegexMatches ("CAPLETTER", Pattern.compile ("[A-Z]")),
						new RegexMatches ("PUNC", Pattern.compile (PUNCTUATION)),
						new RegexMatches ("URL", Pattern.compile ("http\\:.+")),
						new RegexMatches ("EMAIL", Pattern.compile ("URLISH"+"+\\@"+URLISH+"\\."+URLISH+"+")),
						new CoraLayoutFeatures(),
						new TrieLexiconMembership ("DAYMONTH", new File(LEXDIR + "days_months.txt"), false),
						new TrieLexiconMembership ("AFFILIATION", new File(LEXDIR + "affiliation.txt"), false),
						//                  new PrintTokenSequenceFeatures(),
						new TokenSequence2FeatureVectorSequence (true, true),
				});
			}
			// finally, the vanilla case (i.e., all the features come with the data)
			if (VOCAB.equals("email")) {
				pipe = new SerialPipes (new Pipe[] {
						new Vanilla2TokenSequence (),
						new TerminalLines(),
						//                  new PrintTokenSequenceFeatures(),
						new TokenSequence2FeatureVectorSequence (true, true),
				});
			}
		} catch (Exception e) {
			System.err.println("ERROR: "+e);
		}
		return pipe;
	}

	/**
	 * getEvaluator
	 */
	static public MultiSegmentationEvaluator getEvaluator() {
		String[] bTags = new String[]{"B-PROTEIN","B-DNA","B-RNA","B-CELL_LINE","B-CELL_TYPE"};
		String[] iTags = new String[]{"I-PROTEIN","I-DNA","I-RNA","I-CELL_LINE","I-CELL_TYPE"};
		if (VOCAB.equals("ner")) {
			bTags = new String[]{"B-PER","B-ORG","B-LOC","B-MISC"};
			iTags = new String[]{"I-PER","I-ORG","I-LOC","I-MISC"};         
		}
		if (VOCAB.equals("parse")) {
			bTags = new String[]{"B-ADJP", "B-ADVP", "B-CONJP", "B-INTJ", "B-LST", "B-NP", "B-PP", "B-PRT", "B-SBAR",
					"B-UCP", "B-VP"};
			iTags = new String[]{"I-ADJP", "I-ADVP", "I-CONJP", "I-INTJ", "I-LST", "I-NP", "I-PP", "I-PRT", "I-SBAR", 
					"I-UCP", "I-VP"};
		}
		if (VOCAB.equals("cora-headers")) {
			bTags = new String[]{"B-ABSTRACT", "B-ADDRESS", "B-AFFILIATION", "B-AUTHOR", "B-DATE", "B-DEGREE", "B-EMAIL", "B-INTRO", "B-KEYWORD", "B-NOTE", "B-PAGE", "B-PHONE", "B-PUBNUM", "B-TITLE", "B-WEB"};
			iTags = new String[]{"I-ABSTRACT", "I-ADDRESS", "I-AFFILIATION", "I-AUTHOR", "I-DATE", "I-DEGREE", "I-EMAIL", "I-INTRO", "I-KEYWORD", "I-NOTE", "I-PAGE", "I-PHONE", "I-PUBNUM", "I-TITLE", "I-WEB"};
		}
		if (VOCAB.equals("cora-references")) {
			bTags = new String[]{"B-AUTHOR", "B-BOOKTITLE", "B-DATE", "B-EDITOR", "B-INSTITUTION", "B-JOURNAL", "B-LOCATION", "B-NOTE", "B-PAGES", "B-PUBLISHER", "B-TECH", "B-TITLE", "B-VOLUME"};
			iTags = new String[]{"I-AUTHOR", "I-BOOKTITLE", "I-DATE", "I-EDITOR", "I-INSTITUTION", "I-JOURNAL", "I-LOCATION", "I-NOTE", "I-PAGES", "I-PUBLISHER", "I-TECH", "I-TITLE", "I-VOLUME"};
		}
		if (VOCAB.equals("email")) {
			//          bTags = new String[]{"REPLY", "SIG"};
			//          iTags = new String[]{"", ""};
			bTags = new String[]{"B-REPLY", "B-SIG"};
			iTags = new String[]{"I-REPLY", "I-SIG"};
		}
		if (VOCAB.equals("iesig")) {
			bTags = new String[]{"B-CITY","B-COUNTRY","B-EMAIL","B-FAX","B-JOBTITLE","B-NAME","B-ORG","B-PHONE","B-STATE","B-STREET","B-URL","B-ZIP"};
			iTags = new String[]{"I-CITY","I-COUNTRY","I-EMAIL","I-FAX","I-JOBTITLE","I-NAME","I-ORG","I-PHONE","I-STATE","I-STREET","I-URL","I-ZIP"};
		}
		return new MultiSegmentationEvaluator (bTags, iTags, false);
	}

	/**
	 * getIndex
	 */
	public static int getInstanceIndex(Instance inst) {
		String[] name = inst.getName().toString().split(":", 2);
		return Integer.parseInt(name[0]);
	}

	/**
	 * makeIndices
	 */
	public static void makeInstanceIndices(InstanceList ilist) {
		for (int i=0; i<ilist.size(); i++) {
			Instance inst = (Instance)ilist.get(i);
			String newName = i+":"+inst.getName().toString();
			inst.unLock();
			inst.setName(newName);
			inst.setLock();
		}
	}

	public static void computeDensities(InstanceList ilist) {
		densityHash.clear();
		Misc.makeKernelVectors(ilist, null);
		int numFeats = ilist.getDataAlphabet().size();
		for (int i=0; i<ilist.size(); i++) {
			Instance inst1 = (Instance)ilist.get(i);
			for (int j=0; j<=i; j++) {
				Instance inst2 = (Instance)ilist.get(j);
				// which similarity function?
						double sim = 0.0;
						if (QUERY.endsWith("cos"))
							sim = Misc.cosSimilarity((SparseVector)inst1.getData(), (SparseVector)inst2.getData());
						else if (QUERY.endsWith("kl"))
							sim = Misc.klSimilarity((SparseVector)inst1.getData(), (SparseVector)inst2.getData(), numFeats);
						else
							sim = Misc.gaussSimilarity((SparseVector)inst1.getData(), (SparseVector)inst2.getData(), numFeats);
						updateHash(inst1, sim/ilist.size());
						// do it both ways, if different instances
						if (i != j)
							updateHash(inst2, sim/ilist.size());
			}
		}
	}

	/**
	 * updateHash
	 */
	public static void updateHash(Instance inst, double val) {
		if (densityHash.containsKey(inst)) {
			double oldVal = ((Double)densityHash.get(inst)).doubleValue();
			densityHash.put(inst, new Double(val + oldVal));
		}
		else
			densityHash.put(inst, new Double(val));
	}

}