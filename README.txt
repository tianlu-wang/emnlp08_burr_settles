Settles & Craven (EMNLP'08) Source Code
=======================================

Burr Settles
http://www.cs.cmu.edu/~bsettles/

Released: Feb 25, 2011

This code is (approximately) the Java source code used for the active learning
algorithms described and evaluated in the this paper:

	B. Settles and M. Craven. An Analysis of Active Learning Strategies for
	Sequence Labeling Tasks. In Proceedings of the Conference on Empirical 
	Methods in Natural Language Processing (EMNLP), pages 1069-1078. ACL
	Press, 2008.

It is released as-is, with no warranty or guarantees that it will even compile
(It works for me, but I haven't really touched it in years). This code relies
on the MALLET 0.4 implementation of linear-chain CRFs, which is now deprecated
(with the release of MALLET 2.0), and parts of the code rely on my own
modifications to MALLET (particularly EntropyQuerier and FisherMatrixQuerier),
so they may not work for you without some code-wrangling elsewhere. Sorry. :(

PLEASE NOTE: I cannot support or answer questions about the code at this time.
(Also, please don't judge my software engeneering abilities by this... it's
definitely "research-grade," and I've improved over 3 years!) A more formal
and modern package of these algorithms is under development. In the meantime,
I hope you find this useful in your work, as I receive many requests for it!

--Burr



Installation
------------

1. Make sure you have MALLET 0.4 installed. This is an old version of the
MALLET library, available here but no longer maintained:
http://mallet.cs.umass.edu/download.php

2. Edit the 'MALLET_DIR' variable in the Makefile accordingly.

3. Run 'make jar' to compile (and pray).

4. 'runseqall.pl' is a wrapper script, which assumes you have the datasets in
the expected format (see, e.g., NER2TokenSequence.java).

... Most likely, you'll not use the driver programs here, instead invoking
classes from the 'emnlp08.seqactive' package in your own software.



License
-------

This code is arbitrarily released under the GPL v3.0:
http://www.gnu.org/licenses/gpl.html