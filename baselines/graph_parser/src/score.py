# -*- coding: utf-8; -*-

from convert import write_starsem, read_negations

import tempfile;
import subprocess;
import sys;
import os;
import io;

__author__ = "oe"
__version__ = "2018"

def starsem_score(gold, system, file = sys.stdout):
    one = tempfile.mkstemp(prefix = ".sem.score.");
    two = tempfile.mkstemp(prefix = ".sem.score.");
    with open(one[0], "w") as stream:
        write_starsem(gold, stream = stream);
    with open(two[0], "w") as stream:
        write_starsem(system, stream = stream);
    command = ["perl", "eval.cd-sco.pl", "-g", one[1], "-s", two[1]];
    with file if isinstance(file, io.IOBase) else open(file, "w") as stream:
        subprocess.run(command, stdout = stream);
    os.unlink(one[1]);
    os.unlink(two[1]);

if __name__ == "__main__":
    starsem_score(read_negations(sys.argv[1]), read_negations(sys.argv[2]))
