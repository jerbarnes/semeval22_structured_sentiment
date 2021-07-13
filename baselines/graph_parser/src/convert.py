# -*- coding: utf-8; -*-

import argparse;
import sys;
import csv;
import copy;
import json;
from pathlib import Path;

__author__ = "oe"
__version__ = "2018"


def write_starsem(sentences, stream = sys.stdout):
    for sentence in sentences:
      source = sentence["source"] if "source" in sentence else "";
      id = sentence["id"] if "id" in sentence else "-1";
      for node in sentence["nodes"]:
        properties = node["properties"];
        print("{}\t{}\t{}\t{}\t{}\t{}\t_".format(source, id,
                                                 node["id"], node["form"],
                                                 properties["lemma"],
                                                 properties["xpos"]),
              end = "", file = stream);
        negation = node["negation"] if "negation" in node else [];
        if len(negation):
          for instance in negation:
            print("\t{}\t{}\t{}".format(instance.get("cue", "_"),
                                        instance.get("scope", "_"),
                                        instance.get("event", "_")),
                  end = "", file = stream);
          print("", file = stream);
        else:
          print("\t_\t_\t_" if sentence["negations"] else "\t***",
                file = stream);
      print("", file = stream);

def write_epe(sentences, stream = sys.stdout):
    for sentence in sentences:
      print(json.dumps(sentence), file = stream);

def write_tt(sentences, stream = sys.stdout):
    for sentence in sentences:
      for node in sentence["nodes"]:
        form = node["form"];
        negation = node.get("negation");
        if negation and len(negation):
          negation = negation[0];
        if not negation:
          label = "T";
        elif negation.get("cue") == form:
          label = "C";
        elif "cue" in negation:
          label = "A";
        elif "scope" in negation:
          label = "F";
        else:
          label = "T";
        print("{}\t{}".format(form, label), file = stream);
      print(file = stream);

def write_negations(sentences, file = None, format = None):
    if not file and not format:
      format = "epe";
    if file:
      file = Path(file);
      if not format:
        format = file.suffix[1:];
    with file.open(mode = "w") if file else sys.stdout as stream:
      if format in ("*sem", "tsv", "txt"):
        write_starsem(sentences, stream = stream);
      elif format == "epe":
        write_epe(sentences, stream = stream);
      elif format == "tt":
        write_tt(sentences, stream = stream);

def read_starsem(stream):
    sentences = [];
    n = 0;
    reader = csv.reader(stream, delimiter="\t");
    nodes = [];
    for row in reader:
      if len(row):
        source = row[0];
        id = row[1];
        properties = dict(zip(["lemma", "xpos"], (row[4], row[5])));
        node = dict(zip(["id", "form", "properties"],
                        (row[2], row[3], properties)));
        if len(row) > 8:
          negations = [];
          for i in range(0, (len(row) - 7) // 3):
            negation = {"id": n + i};
            i *= 3;
            if row[7 + i] != "_":
              negation["cue"] = row[7 + i];
            if row[7 + i + 1] != "_":
              negation["scope"] = row[7 + i + 1];
            if row[7 + i + 2] != "_":
              negation["event"] = row[7 + i + 2];
            negations.append(negation);
          node["negation"] = negations;
        nodes.append(node);
      else:
        i = len(nodes[0]["negation"]) if "negation" in nodes[0] else 0;
        n += i;
        sentences.append({"id": id, "source": source,
                          "negations": i, "nodes": nodes[:]});
        nodes.clear();

    return sentences;

def read_epe(stream):
    sentences = [];
    for line in stream:
      sentence = json.loads(line);
      i = 0;
      for node in sentence["nodes"]:
        i = max(i, len(node.get("negation", [])));
      sentence["negations"] = i;
      sentences.append(sentence);

    return sentences

def read_negations(file):
    file = Path(file);
    if file.exists():
      suffix = file.suffix[1:];
      with file.open() as stream:
        if suffix in ("*sem", "tsv", "txt"):
          return read_starsem(stream);
        elif suffix == "epe":
          return read_epe(stream);

def distribute_negation_instances(sentences, filter = False):

    instances = []

    for sentence in sentences:
      n = sentence.get("negations", 0);
      for i in range(0, n if filter else max(n, 1)):
        instance = copy.deepcopy(sentence);
        instance["negations"] = 1 if n >= 1 else 0;
        for node in instance["nodes"]:
          if "negation" in node:
            node["negation"] = node["negation"][i:i + 1];
        instances.append(instance);

    return instances;

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Format conversion between *SEM 2012 and EPE 2017 formats");
    parser.add_argument("--input", required = True);
    parser.add_argument("--output");
    parser.add_argument("--format");
    parser.add_argument("--distribute", action="store_true");
    parser.add_argument("--filter", action="store_true");
    arguments = parser.parse_args();

    sentences = read_negations(arguments.input);

    if arguments.distribute:
      sentences \
        = distribute_negation_instances(sentences, filter = arguments.filter);

    write_negations(sentences, arguments.output, format = arguments.format);
