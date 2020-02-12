from SeqIterator import Constants
import gzip

"""
@author: Jacob Porter
@summary: An iterator class for iterating through sequence record files.
@status: A bit kludgy.
"""

class FileTypeError(Exception):
    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg


class SeqReader:
    def __init__(self, file_name, file_type='fasta', gzip_switch = False):
        self.my_init(file_name, file_type=file_type, gzip_switch = gzip_switch)

    def my_init(self, file_name, file_type='fasta', gzip_switch = False):
        self.c = 0
        self.file_name = file_name
        if gzip_switch:
            self.seq_file = gzip.open(file_name, 'r')
        else:
            self.seq_file = open(file_name, 'r')
        self.next_line = self.seq_file.readline()
        self.file_type = file_type.lower()
        if self.file_type=='fasta':
            self.type=0
        elif self.file_type=='fastq':
            self.type=1
        elif self.file_type=='sam':
            self.type=2
            nextline = self.next_line
            while nextline.startswith('@'):
                nextline = self.seq_file.readline()
            self.next_line = nextline
        else:
            raise FileTypeError(file_type, "Unsupported file type.  Check spelling.")

    def __iter__(self):
        return self

    def __next__(self):
        return next(self)

    def __next__(self):
        if self.type == 0:  # FASTA
            seq_id = self.next_line
            seq_seq = self.seq_file.readline()
            if seq_id == None or seq_id == '' or seq_seq == None:
                raise StopIteration
            else:
                self.c += 1
                seq_seq = seq_seq.strip("\n")
                my_line = self.seq_file.readline()
                while my_line != None and my_line != "" and my_line[0] != ">":
                    seq_seq += my_line.strip("\n")
                    my_line = self.seq_file.readline()
                self.next_line = my_line
                return (seq_id.strip(">\n"), seq_seq)
        elif self.type == 1: # FASTQ
            seq_id = self.next_line
            seq_seq = self.seq_file.readline()
            seq_del = self.seq_file.readline()
            seq_qual = self.seq_file.readline()
            if seq_id == None or seq_id == '' or seq_del == None or seq_qual == None:
                raise StopIteration
            self.c += 1
            self.next_line = self.seq_file.readline()
            return (seq_id[1:len(seq_id)].strip("\n"), seq_seq.strip("\n"), seq_qual.strip("\n"), seq_del.strip("\n"))
        elif self.type == 2: # SAM
            sam_dictionary = {}
            sam_line = self.next_line.strip("\n")
            if sam_line == None or sam_line == '':
                raise StopIteration
            self.c += 1
            sam_list = sam_line.split()
            sam_dictionary["QNAME"] = sam_list[0]
            sam_dictionary["FLAG"] = sam_list[1]
            sam_dictionary["RNAME"] = sam_list[2]
            sam_dictionary["POS"] = sam_list[3]
            sam_dictionary["MAPQ"] = sam_list[4]
            sam_dictionary["CIGAR"] = sam_list[5]
            sam_dictionary["RNEXT"] = sam_list[6]
            sam_dictionary["PNEXT"] = sam_list[7]
            sam_dictionary["TLEN"] = sam_list[8]
            sam_dictionary["SEQ"] = sam_list[9]
            sam_dictionary["QUAL"] = sam_list[10]
            for i in range(10,len(sam_list)):
                if sam_list[i].startswith("AS:i"):
                    sam_dictionary[Constants.SAM_KEY_ALIGNMENT_SCORE] = sam_list[i].strip("AS:i:")
                elif sam_list[i].startswith("AS:f"):
                    sam_dictionary[Constants.SAM_KEY_ALIGNMENT_SCORE] = sam_list[i].strip("AS:f:")
                elif sam_list[i].startswith("NM:i"):
                    sam_dictionary["NM:i"] = sam_list[i].strip("NM:i:")
                elif sam_list[i].startswith("NH:i"):
                    sam_dictionary["NH:i"] = sam_list[i].strip("NH:i:")
                elif sam_list[i].startswith("IH:i"):
                    sam_dictionary["IH:i"] = sam_list[i].strip("IH:i:")
                elif sam_list[i].startswith("HI:i"):
                    sam_dictionary["HI:i"] = sam_list[i].strip("HI:i:")
                elif sam_list[i].startswith("MD:Z"):
                    sam_dictionary["MD:Z"] = sam_list[i].strip("MD:Z:")
                elif sam_list[i].startswith("XA:i"):
                    sam_dictionary["XA:i"] = sam_list[i].strip("XA:i:")
                elif sam_list[i].startswith("XA:Z"):
                    sam_dictionary["XA:Z"] = sam_list[i].strip("XA:Z:")
                elif sam_list[i].startswith("XS:Z"):
                    sam_dictionary["XS:Z"] = sam_list[i].strip("XS:Z:")
            self.next_line = self.seq_file.readline()
            return sam_dictionary
        else: # Unsupported
            raise StopIteration

    def close(self):
        self.seq_file.close()

    def reset(self):
        self.c = 0
        self.seq_file.close()
        self.my_init(self.file_name, file_type=self.file_type)

    def count(self):
        count = 0
        for _ in self:
            count += 1
        self.reset()
        return count

    def peekAtId(self):
        if self.next_line == None or self.next_line == '':
            return None
        if self.type == 0:
            return self.next_line.strip(">\n")
        elif self.type == 1:
            return self.next_line[1:len(self.next_line)].strip("\n")
        elif self.type == 2:
            return self.next_line.strip("\n").split()[0]

    def records_processed(self):
        return self.c

    def convertToDict(self, *argv):
        record_dict = {}
        #my_counter = 0
        for SAM_record in self:
            QNAME = SAM_record[Constants.SAM_KEY_QNAME]
            for s in argv:
                if QNAME.endswith(s):
                    QNAME = QNAME[0:len(QNAME) - len(s)]
                    break
            record_list = record_dict.get(QNAME, [])
            record_list.append(SAM_record)
            record_dict[QNAME] = record_list
            #my_counter += 1
        return record_dict


class SeqWriter:

    sam_standard_fields = ["QNAME", "FLAG", "RNAME", "POS", "MAPQ", "CIGAR", "RNEXT", "PNEXT", "TLEN", "SEQ", "QUAL"]
    bfast_fields = ["PG:Z", Constants.SAM_KEY_ALIGNMENT_SCORE, "NM:i", "NH:i", "IH:i", "HI:i", "MD:Z", "XA:i", "XR:Z", "XG:Z", "XM:Z", Constants.SAM_KEY_RECOVERED, Constants.SAM_KEY_RESCORE]

    def __init__(self, seq_file, file_type='fasta', line_toggle=False, line_length=80):
        self.seq_file = seq_file
        self.current = 0
        self.line_toggle = line_toggle
        self.line_length = line_length
        file_type = file_type.lower()
        if file_type=='fasta':
            self.type=0
        elif file_type=='fastq':
            self.type=1
        elif file_type=='sam':
            self.type=2
        else:
            raise FileTypeError(file_type, "Unsupported file type.  Check spelling.")

    def write(self, seq):
        if self.type == 0: #Fasta
            if self.line_toggle:
                self.seq_file.write(">" + seq[0] + "\n")
                count = 0
                seq_str = seq[1]
                while count + self.line_length < len(seq_str):
                    seq_out = seq_str[count:count+self.line_length]
                    self.seq_file.write(seq_out + "\n")
                    count += self.line_length
                seq_out = seq_str[count:len(seq_str)]
                self.seq_file.write(seq_out + "\n")
            else:
                self.seq_file.write(">" + seq[0] + "\n" + seq[1] + "\n")
        elif self.type == 1: #Fastq
            separator = "+"
            if len(seq) == 4:
                separator = seq[3]
            self.seq_file.write("@" + seq[0] + "\n" + seq[1] + "\n" + separator + "\n" + seq[2] + "\n")
        elif self.type == 2: #SAM
            if isinstance(seq, str): #For writing comments
                self.seq_file.write(seq)
            else: #Write a record
                line_string = ""
                for field in SeqWriter.sam_standard_fields:
                    line_string += seq[field] + "\t"
                for field in seq: #Could use complementation on the list of keys instead
                    if field not in SeqWriter.sam_standard_fields:
                        value = str(seq[field])
                        if (field == Constants.SAM_KEY_ALIGNMENT_SCORE):
                            field = Constants.SAM_KEY_ALIGNMENT_SCORE + ":f"
                        line_string += field + ":" + value + "\t"
                self.seq_file.write(line_string + "\n")

    def close(self):
        self.seq_file.close()

    def flush(self):
        self.seq_file.flush()

