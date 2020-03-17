#!/usr/bin/env python
# title             :dvf.py
# description       :Identifying viral sequences from metagenomic data by deep learning
# author            :Jie Ren renj@usc.edu
# date              :20180807
# version           :1.0
# usage             :python dvf.py -i <path_to_input_fasta> -o <path_to_output_directory>
# required packages :numpy, tensorflow, keras
# conda create -n dvf python=3.6 numpy tensorflow-gpu keras-gpu scikit-learn Biopython
#==============================================================================

import datetime
import multiprocessing
import optparse
#### Step 0: pass arguments into the program ####
import os
import sys
import warnings
from collections import namedtuple

#### Step 0: import keras libraries ####
import h5py
# os.environ['KERAS_BACKEND'] = 'theano'
import keras
import numpy as np
from keras.models import load_model
from tqdm import tqdm

from SeqIterator.SeqIterator import SeqReader, SeqWriter

tick = datetime.datetime.now()
print("Started on : {}.".format(tick), file=sys.stdout)
sys.stdout.flush()

prog_base = os.path.split(sys.argv[0])[1]
parser = optparse.OptionParser()
parser.add_option("-i",
                  "--in",
                  action="store",
                  type="string",
                  dest="input_fa",
                  help="input fasta file")
parser.add_option("-m",
                  "--mod",
                  action="store",
                  type="string",
                  dest="modDir",
                  default=os.path.join(
                      os.path.dirname(os.path.abspath(__file__)), "models"),
                  help="model directory (default ./models)")
parser.add_option("-o",
                  "--out",
                  action="store",
                  type="string",
                  dest="output_dir",
                  default='./',
                  help="output directory")
parser.add_option("-b",
                  "--batch_size",
                  type=int,
                  help="The number of sequences to predict at once.",
                  default=256)
# parser.add_option("-l",
#                   "--len",
#                   action="store",
#                   type="int",
#                   dest="cutoff_len",
#                   default=1,
#                   help="predict only for sequence >= L bp (default 1)")
# parser.add_option("-c",
#                   "--core",
#                   action="store",
#                   type="int",
#                   dest="core_num",
#                   default=1,
#                   help="number of parallel cores (default 1)")

(options, args) = parser.parse_args()
if (options.input_fa is None):
    sys.stderr.write(prog_base +
                     ": ERROR: missing required command-line argument")
    filelog.write(prog_base +
                  ": ERROR: missing required command-line argument")
    parser.print_help()
    sys.exit(0)
batch_size = options.batch_size
input_fa = options.input_fa
if options.output_dir != './':
    output_dir = options.output_dir
else:
    output_dir = os.path.dirname(os.path.abspath(input_fa))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# cutoff_len = options.cutoff_len
# core_num = options.core_num

#sys.setrecursionlimit(10000000)
#os.environ['THEANO_FLAGS'] = "floatX=float32,openmp=True"
#os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=gpu0,floatX=float32"
#os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())


#### Step 0: function for encoding sequences into matrices of size 4 by n ####
def encodeSeq(seq):
    seq_code = list()
    for pos in range(len(seq)):
        letter = seq[pos]
        if letter in ['A', 'a']:
            code = [1, 0, 0, 0]
        elif letter in ['C', 'c']:
            code = [0, 1, 0, 0]
        elif letter in ['G', 'g']:
            code = [0, 0, 1, 0]
        elif letter in ['T', 't']:
            code = [0, 0, 0, 1]
        else:
            code = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
        seq_code.append(code)
    return seq_code


complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}


#### Step 1: load model ####
print("1. Loading Models.")
#modDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
modDir = options.modDir

print("   model directory {}".format(modDir))

modDict = {}
nullDict = {}

warnings.filterwarnings('ignore', 'Error in loading the saved optimizer ')

# for contigLengthk in ['0.15', '0.3', '0.5', '1'] :
for contigLengthk in ['0.3', '0.5', '1']:
    modPattern = 'model_siamese_varlen_' + contigLengthk + 'k'
    try:
        modName = [
            x for x in os.listdir(modDir)
            if modPattern in x and x.endswith(".h5")
        ][0]
    except IndexError:
        print("Skipping: {}".format(contigLengthk), file=sys.stderr)
        continue
    modDict[contigLengthk] = load_model(os.path.join(modDir, modName))
    Y_pred_file = [
        x for x in os.listdir(modDir) if modPattern in x and "Y_pred" in x
    ][0]
    print("Loading: {}".format(contigLengthk), file=sys.stderr)
    with open(os.path.join(modDir, Y_pred_file)) as f:
        tmp = [line.split() for line in f][0]
        Y_pred = [float(x) for x in tmp]
    Y_true_file = [
        x for x in os.listdir(modDir) if modPattern in x and "Y_true" in x
    ][0]
    with open(os.path.join(modDir, Y_true_file)) as f:
        tmp = [line.split()[0] for line in f]
        Y_true = [float(x) for x in tmp]
    nullDict[contigLengthk] = Y_pred[:Y_true.index(1)]
    print("Null dict length: ",
          len(nullDict[contigLengthk]),
          contigLengthk,
          file=sys.stderr)
    sys.stderr.flush()

#### Step2 : encode sequences in input fasta, and predict scores ####

# clean the output file
outfile = os.path.join(output_dir,
                       os.path.basename(input_fa) + '_bp_dvfpred.txt')
predF = open(outfile, 'w')
writef = predF.write('\t'.join(['name', 'len', 'score', 'label']) + '\n')
predF.close()
predF = open(outfile, 'a')
Batch_R = namedtuple('Batch_R', 'fw rv id')


def jsp_pred(batch, model=modDict['0.3']):
    batch_fw = np.array([item.fw for item in batch])
    batch_rv = np.array([item.rv for item in batch])
    score = model.predict([batch_fw, batch_rv], batch_size=len(batch))
    for i in range(len(batch)):
        print("{}\t{}\t{}\t{}".format(batch[i].id, len(batch[i].fw),
                                      float(score[i]),
                                      0 if float(score[i]) < 0.5 else 1),
              file=predF)
    predF.flush()


reader = SeqReader(input_fa)
batch = []
for record in tqdm(reader):
    code_fw = encodeSeq(record[1])
    code_rv = encodeSeq("".join(
        complement.get(base, base) for base in reversed(record[1])))
    batch.append(Batch_R(fw=code_fw, rv=code_rv, id=record[0]))
    if len(batch) == batch_size:
        jsp_pred(batch)
        batch = []
jsp_pred(batch)


tock = datetime.datetime.now()
print("3. Done. Thank you for using DeepVirFinder.")
print("   output in {}".format(outfile))
print("The process took time: {}".format(tock - tick), file=sys.stderr)
sys.stderr.flush()
