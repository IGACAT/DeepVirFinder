"""
@author: Jacob Porter
@summary: This file stores constants and defaults for BisPin.
"""

#File type constants
FASTA = 'fasta'
FASTQ = 'fastq'
SAM = 'sam'

#Aligner constants
BFAST = "BFAST"
BWA = "BWA"

#Program information
version = '0.1.1'
creation_string = "Created by Jacob Porter at Virginia Tech.  www.jacobporter.com"

#Read and reference conversion type combinations
CONV_CT_CT = "CT_CT"
CONV_CT_GA = "CT_GA"
CONV_GA_CT = "GA_CT"
CONV_GA_GA = "GA_GA"
CONV_CT_GA_CT = "CT_GA_CT"
CONV_CT_GA_GA = "CT_GA_GA"
CONV_GA_CT_CT = "GA_CT_CT"
CONV_GA_CT_GA = "GA_CT_GA"
CONV_R_R = "R_R"
CONV_R = "R"
CONV_CT = "CT"
CONV_GA = "GA"
CONV_FORWARD = True
CONV_REVERSE = False
CONV_SEP = "_"
CONV_TYPES = [CONV_CT_CT, CONV_CT_GA, CONV_GA_CT, CONV_GA_GA, CONV_R_R, CONV_CT_GA_CT, CONV_CT_GA_GA, CONV_GA_CT_CT, CONV_GA_CT_GA]

#Conversion descriptions for the final report.
CONV_DESC = {CONV_CT_CT:"+FW original top (OT)", CONV_CT_GA:"-FW original bottom (OB)", CONV_GA_CT:"+RC complementary to top strand (CTOT)", CONV_GA_GA:"-RC complementary to bottom strand (CTOB)",
             CONV_R_R:"Recovered untreated strand", CONV_CT_GA_CT:"+FW original top (OT)", CONV_CT_GA_GA:"-FW original bottom (OB)", CONV_GA_CT_CT:"+RC complementary to top strand (CTOT)", CONV_GA_CT_GA:"-RC complementary to bottom strand (CTOB)"}

#SAM record keys
SAM_KEY_QNAME = "QNAME"
SAM_KEY_FLAG = "FLAG"
SAM_KEY_RNAME = "RNAME"
SAM_KEY_POS = "POS"
SAM_KEY_MAPQ = "MAPQ"
SAM_KEY_CIGAR = "CIGAR"
SAM_KEY_RNEXT = "RNEXT"
SAM_KEY_PNEXT = "PNEXT"
SAM_KEY_TLEN = "TLEN"
SAM_KEY_SEQ = "SEQ"
SAM_KEY_QUAL = "QUAL"
SAM_KEY_ALIGNMENT_SCORE = "AS"
SAM_KEY_MD = "MD:Z"
SAM_KEY_PROGRAM = "PG:Z"
SAM_KEY_DISTANCE = "NM:i"
SAM_KEY_READ = "XR:Z"
SAM_KEY_GENOME = "XG:Z"
SAM_KEY_METHYLATION = "XM:Z"
SAM_KEY_METHYLATION_CT = "XM:C"
SAM_KEY_METHYLATION_GA = "XM:G"
SAM_KEY_RECOVERED = "XV:Z"
SAM_KEY_RESCORE = "XS:Z"

SAM_KEYS_TO_KEEP = [SAM_KEY_QNAME, SAM_KEY_FLAG, SAM_KEY_RNAME, SAM_KEY_POS, SAM_KEY_MAPQ,
                    SAM_KEY_CIGAR, SAM_KEY_RNEXT, SAM_KEY_PNEXT, SAM_KEY_TLEN, SAM_KEY_SEQ,
                    SAM_KEY_QUAL, SAM_KEY_ALIGNMENT_SCORE, SAM_KEY_MD,
                    SAM_KEY_READ, SAM_KEY_GENOME]

SAM_KEYS_EXTRA = [SAM_KEY_ALIGNMENT_SCORE, SAM_KEY_MD, SAM_KEY_PROGRAM, SAM_KEY_DISTANCE,
                  SAM_KEY_READ, SAM_KEY_GENOME, SAM_KEY_METHYLATION, SAM_KEY_RECOVERED,
                  SAM_KEY_RESCORE]

#SAM record values
SAM_VALUE_UNMAPPED = "4"
SAM_VALUE_FILTERED = "512"
SAM_VALUE_SECONDARY = 256
SAM_VALUE_STAR = "*"
SAM_VALUE_ZERO = "0"
SAM_VALUE_PROGRAM = "BisPin"
SAM_VALUE_BFAST_UNMAPPED_SCORE = "-2147483648"

#Record write type for postprocessing the SAM file
SAM_WRITE_ALIGNED = 0
SAM_WRITE_UNIQUE = 1
SAM_WRITE_AMBIGUOUS = 2
SAM_WRITE_UNMAPPED = 3
SAM_WRITE_FILTERED = 4
SAM_WRITE_STATS = 5

#Layout
LAYOUT = "LAYOUT"
LAYOUT_SINGLE = 0
LAYOUT_PAIRED = 1

#Construction and mapping protocol
PROTOCOL = "PROTOCOL"
PROTOCOL_DIRECTIONAL = 0
PROTOCOL_BIDIRECTIONAL = 1
PROTOCOL_PBAT = 2
PROTOCOL_HAIRPIN = 3
PROTOCOL_ONLYCT = 4
PROTOCOL_ONLYGA = 5

#Post process argument names:
BISPIN_READS1 = "READS1"
BISPIN_READS2 = "READS2"
BISPIN_CUTOFF_BS = "CutoffBS"
BISPIN_CUTOFF_R = "CutoffR"
BISPIN_AREADS = "AREADS"
BISPIN_TREADS = "TREADS"
BISPIN_RESCORE_MATRIX = "Matrix"

#Cigar Token Strings
CIGAR_I = "I"
CIGAR_M = "M"
CIGAR_D = "D"

#Rescore Dictionary Key Values
OPEN_DEL = "open_d"
OPEN_INS = "open_i"
EXT_DEL = "ext_d"
EXT_INS = "ext_i"

#Default values for the scoring functions for the BFAST alignment
#Alignment scoring function
MATCH_SCORE = 96
MISMATCH_SCORE = -90
GAP_OPEN = -400
GAP_EXTENSION = -30

#Old values
#GAP_OPEN = -50
#GAP_EXTENSION = -30
#MATCH_SCORE = 20
#MISMATCH_SCORE = -40

#The prefix for the alignment scoring function file name
ALIGNFILENAME = 'alignment_scoring_function_'
HAIRPINFILEPREFIX = "BisPin.hairpin."

#Score filter threshold values
BISPIN_DEFAULT_CUTOFF_BS = 45
BISPIN_DEFAULT_CUTOFF_R = 40

#Rescoring function
#(F Chiaromonte, VB Yap, W Miller, PSB 2002:115-126). blastz default matrix
#The row is the reference genome and the column is the read or pattern string
#The last line is the gap extension penalty and it is in the order deletions, insertions
#The penultimate line is the gap open penalty and it is in the order deletions, insertions
RESCORE_DEFAULT = """
     A    C    G    T    N
A   91 -114  -31 -123 -100
C -114  100 -125  -31 -100
G  -31 -125  100 -114 -100
T -123  -31 -114   91 -100
N -100 -100 -100 -100 100
-400, -400
-30, -30 """

#Amount of random characters to create for the filename of a temporary file
BISPIN_RANDOM_LENGTH = 10

#File input convenience features and BFAST parameter names
GZIP = "GZIP"
STARTREADNUM = "STARTREADNUM"
ENDREADNUM = "ENDREADNUM"
DEFAULTNUMPROCESSES = 4
SINGLEPROCESSPOST = "SINGLEPROCESSPOST"
keySize = "keySize"
maxKeyMatches = "maxKeyMatches"
maxNumReadMatches = "maxNumReadMatches"
maxNumAlignmentMatches = "maxNumAlignmentMatches"
insertSizeAvg = "insertSizeAvg"
insertSizeStdDev = "insertSizeStdDev"
mainIndexes = "mainIndexes"
secondaryIndexes = "secondaryIndexes"
offsets = "offsets"
USESECONDARYINDEXES = "use_secondary_indexes"
PAIREDEXPRESSION = "PAIREDEXPRESSION"
HAIRPINPOSTFILE = "HAIRPINPOSTFILE"
REMOVECOMMENTS = "REMOVECOMMENTS"
HAIRPIN_PAIRED = "Hairpin_Paired"
