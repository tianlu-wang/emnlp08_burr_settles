package emnlp08.pipes;

// burr settles

import edu.umass.cs.mallet.base.pipe.*;
import edu.umass.cs.mallet.base.types.TokenSequence;
import edu.umass.cs.mallet.base.types.Token;
import edu.umass.cs.mallet.base.types.Instance;
import java.util.HashMap;
import java.io.*;

public class FixName extends Pipe implements Serializable
{

	public Instance pipe (Instance carrier) {
		String[] arr = carrier.getName().toString().split("/");
		String name = arr[arr.length-2]+"/"+arr[arr.length-1];
		carrier.setName(name);
		return carrier;
	}

	// Serialization 

	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;

	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);
	}

	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int version = in.readInt ();
	}
}
