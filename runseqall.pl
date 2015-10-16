#!/usr/bin/perl -w

if (!@ARGV) {
	die "Usage: % runall.pl <files...>\n";
}

$dirs = join(" ", @ARGV);

$data = $ARGV[0];
$data =~ s/^.*\///;
$data =~ s/\.(.*?)$//;

$java = "java -Xmx3500m -cp .:/Users/bsettles/Research/lib/mallet-0.4/lib/mallet-deps.jar:/Users/bsettles/Research/lib/mallet-0.4/class/:./emnlp08.jar";

@modes = ("rand", "longest", "leastconf", "margin", "ent", "ent-kbest", "ent-tok", "ent-avgtok", "kde-cos", "qbc-ve", "qbc-kl", "fisher");

@modes = ("longest");

$opts = "";
if ($data eq "reuters") {
	$opts = " -v ner ";
}
if ($data eq "parsing") {
	$opts = " -v parse ";
}
if ($data eq "headers") {
	$opts = " -v cora-headers ";
}
if ($data eq "references") {
	$opts = " -v cora-references ";
}
if ($data eq "sigplusreply") {
	# $opts = " -i 1 -b 1 -n 50 -v email ";
	$opts = " -v email ";
}
if ($data eq "iesig") {
	$opts = " -v iesig ";
}

foreach $m (@modes) {
	$outfile = "results/$data.$m.out";
	if (-e $outfile) {
		print STDERR "!!!! ALREADY EXISTS: $outfile\n";
	}
	else {	
		# print "$java emnlp08.SequenceExp -q $m $vocab $dirs > $outfile\n";
		system "$java emnlp08.SequenceExp -f 5 -q $m $opts $dirs > $outfile";
	}
}