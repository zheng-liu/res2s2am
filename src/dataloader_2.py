import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class GenomeDataset(Dataset):
    """ GWAS and nonGWAS SNP sequence dataset"""

    def __init__(self, filename, seq_name=['seq_ref_1'], encode_mode="N_2_zero", metadata_cols=None):
        self.genomedata = pd.read_csv(filename, sep="\t")
        self.seq_name = seq_name
        self.encode_mode = encode_mode
        self.metadata_cols = metadata_cols

    def __len__(self):
        return len(self.genomedata)

    def __getitem__(self, idx):
        seq_encode = {sn: self.encode(self.genomedata.ix[idx, sn], self.encode_mode) for sn in self.seq_name}
        label = self.genomedata.ix[idx, "label"]
        rsid = self.genomedata.ix[idx, 'rsid']
        query_snp_rsid = self.genomedata.ix[idx, 'query_snp_rsid']
        if self.metadata_cols is not None:
            metadata = self.encode_metadata(self.genomedata.ix[idx, self.metadata_cols])
            seq_tuple = tuple([seq_encode[sn] for sn in self.seq_name] + [label] + [rsid] + [query_snp_rsid] + [metadata])
        else:
            seq_tuple = tuple([seq_encode[sn] for sn in self.seq_name] + [label] + [rsid] + [query_snp_rsid])
        return seq_tuple


    def encode(self, input, encode_mode='N_2_zero'):
        """ Encode string input to a numerical matrix. Sequence after encoding has two modes:
            N_2_zero: "N" encodes to [0,0,0,0]
            N_2_quarter: "N" encodes to [1/4, 1/4, 1/4, 1/4]
        """

        if encode_mode == "N_2_zero":
            # output 1*4*n numpy binary matrix in "A, C, G, T" order
            # nucleotide "N" encoded as [0, 0, 0, 0]
            n = len(input)
            output = np.zeros((4, n), dtype="f")
            for i in range(n):
                if input[i] == "A" or input[i] == "a":
                    output[0, i] = 1.0
                elif input[i] == "C" or input[i] == "c":
                    output[1, i] = 1.0
                elif input[i] == "G" or input[i] == "g":
                    output[2, i] = 1.0
                elif input[i] == "T" or input[i] == "t":
                    output[3, i] = 1.0
                else:
                    pass

        elif encode_mode == "N_2_quarter":
            # output 1*4*n numpy integer matrix in "A, C, G, T" order
            # nucleotide "N" encoded as [1/4, 1/4, 1/4, 1/4]
            n = len(input)
            output = np.zeros((4, n), dtype="f")
            for i in range(n):
                if input[i] == "A" or input[i] == "a":
                    output[0, i] = 1.0
                elif input[i] == "C" or input[i] == "c":
                    output[1, i] = 2.0
                elif input[i] == "G" or input[i] == "g":
                    output[2, i] = 3.0
                elif input[i] == "T" or input[i] == "t":
                    output[3, i] = 4.0
                else:
                    output[0, i] = 0.25
                    output[1, i] = 0.25
                    output[2, i] = 0.25
                    output[3, i] = 0.25

        return output

    def encode_metadata(self, inputs):
        n = len(inputs)
        output = np.zeros((1, n), dtype='f')
        for i in range(n):
            if inputs[i] is np.nan:
                output[0, i] = -1.0
            else:
                output[0, i] = inputs[i]
        return output

if __name__ == '__main__':
    filename = 'tmp.txt'
    data = GenomeDataset(filename, has_metadata=True)
    print(data.genomedata)
    # for d in data:
    #     print(d)
        #break
