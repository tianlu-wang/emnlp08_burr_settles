EMNLP08_DIR = $(shell pwd)
MALLET_DIR = /Users/bsettles/Research/lib/mallet-0.4

####
# compiler and flag variables... you may need to edit these.

JAVA = java
JAVAC = javac
JAVA_FLAGS = \
-classpath ".:$(MALLET_DIR)/lib/mallet-deps.jar:$(MALLET_DIR)/class" \
-g:lines,vars,source \
-d $(EMNLP08_DIR)/bin \
-J-Xmx200m

JAVADOC = javadoc
JAVADOC_FLAGS = -J-Xmx300m

####
# the rules

all: bin
	$(JAVAC) $(JAVA_FLAGS) `find src -name '*.java'`

jar: all
	jar -cvf emnlp08.jar -C bin emnlp08

javadoc: html bin
	$(JAVADOC) $(JAVADOC_FLAGS) -classpath "$(EMNLP08_DIR)/bin" -d $(EMNLP08_DIR)/javadoc -sourcepath $(EMNLP08_DIR)/src -source 1.4 -subpackages emnlp08

bin:
	mkdir -p bin

html:
	mkdir -p javadoc

####
# remove all generated files

clean:
	rm -rf *.jar bin/ javadoc/