import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.pool import NullPool
# import ast

class annotate():

    def __init__(self, snp_df_input, coord_version):
        if coord_version == 'hg19':
            self.db_url = {
                "drivername": 'mysql+pymysql',
                "host": "genome-mysql.cse.ucsc.edu",
                "port": "3306",
                "username": "genome",
                "password": "",
                "database": 'hg19',
                "query": {'charset': 'utf8'}
            }
            query = '''
                    SELECT chrom, strand, cdsStart, cdsEnd, exonStarts, exonEnds, txStart, txEnd
                    FROM knownGene
                    '''

        elif coord_version == 'hg38':
            self.db_url = {
                "drivername": 'mysql+pymysql',
                "host": "genome-mysql.cse.ucsc.edu",
                "port": "3306",
                "username": "genome",
                "password": "",
                "database": 'hg38',
                "query": {'charset': 'utf8'}
            }
            query = '''
                    SELECT chrom, strand, cdsStart, cdsEnd, exonStarts, exonEnds, txStart, txEnd
                    FROM knownGene
                    '''

        else:
            print('[ERROR] coordinate system error!')
            sys.exit()


        self.db = create_engine(URL(**self.db_url), poolclass=NullPool, pool_recycle=3600)
        self.cores = mp.cpu_count()
        self.snp_df_input = snp_df_input

        rows = self.db.execute(query)
        self.GeneDB = pd.DataFrame(rows.fetchall(), columns=["chrom", "strand", "cdsStart", "cdsEnd", "exonStarts", "exonEnds", "txStart", "txEnd"])
        self.GeneDB["exonStarts"] = self.GeneDB.apply(lambda x: [int(es) for es in x["exonStarts"].decode("utf-8")[:-1].split(",")], axis=1)
        self.GeneDB["exonEnds"] = self.GeneDB.apply(lambda x: [int(ee) for ee in x["exonEnds"].decode("utf-8")[:-1].split(",")], axis=1)

        if coord_version == 'hg19':
            print("[INFO] hg19 ensGene database Loading Successfully!")
        elif coord_version == 'hg38':
            print("[INFO] hg38 knownGene database Loading Successfully!")

    def __exit__(self):
        self.db.dispose()

    def annotate_snp(self, chrom, chromStart):
        transcript = self.GeneDB.loc[self.GeneDB["chrom"] == chrom]
        for strand, txStart, txEnd, cdsStart, cdsEnd, exonStarts, exonEnds in zip(transcript["strand"], transcript["txStart"], transcript["txEnd"], \
            transcript["cdsStart"], transcript["cdsEnd"], transcript["exonStarts"], transcript["exonEnds"]):

            if txStart <= chromStart <= txEnd:
                # "+" strand
                if strand == "+":
                    if cdsStart != cdsEnd:
                        if txStart <= chromStart <= cdsStart and exonStarts[0] <= cdsStart <= exonEnds[0]:
                            return "5putr"
                        if cdsEnd <= chromStart <= txEnd and exonStarts[-1] <= cdsEnd <= exonEnds[-1]:
                            return "3putr"
                        for exon_start, exon_end in zip(exonStarts, exonEnds):
                            if exon_start <= chromStart <= exon_end:
                                return "pcexon"
                        return "intron"
                    if cdsStart == cdsEnd:
                        for exon_start, exon_end in zip(exonStarts, exonEnds):
                            if exon_start <= chromStart <= exon_end:
                                return "nonpcexon"
                        return "intron"

                # "-" strand
                else:
                    if cdsStart != cdsEnd:
                        if txStart <= chromStart <= cdsStart and exonStarts[0] <= cdsStart <= exonEnds[0]:
                            return "3putr"
                        if cdsEnd <= chromStart <= txEnd and exonStarts[-1] <= cdsEnd <= exonEnds[-1]:
                            return "5putr"
                        for exon_start, exon_end in zip(exonStarts, exonEnds):
                            if exon_start <= chromStart <= exon_end:
                                return "pcexon"
                        return "intron"
                    if cdsStart == cdsEnd:
                        for exon_start, exon_end in zip(exonStarts, exonEnds):
                            if exon_start <= chromStart <= exon_end:
                                return "nonpcexon"
                        return "intron"

        return "intergenic"

    def annotate_snp_list(self, snp_df):
        tqdm.pandas()
        snp_df = snp_df.copy()
        snp_df["annotation"] = ""
        snp_df.loc[:, "annotation"] = snp_df.progress_apply(lambda x: self.annotate_snp(x["chr"], x["pos"]), axis=1)
        self.annot_list.append(snp_df)

    def fast_annotate_snp_list(self):
        self.manager = mp.Manager()
        self.annot_list = self.manager.list()
        snp_df_list = np.array_split(self.snp_df_input, self.cores)

        processes = []
        for snp_df in snp_df_list:
            p = mp.Process(target=self.annotate_snp_list, args=(snp_df,))
            processes.append(p)
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        self.grasp_w_annotation = pd.concat(self.annot_list)
        return self.grasp_w_annotation

if __name__ == "__main__":
    grasp_sub = pd.read_csv("tmpx.txt", sep="\t", error_bad_lines=False).rename(columns={'chr':'chr(hg19)', 'pos_hg38': 'pos(hg19)'})
    print(grasp_sub)
    annotate_ob = annotate(grasp_sub, coord_version='hg38')
    grasp_sub_w_annot = annotate_ob.fast_annotate_snp_list()
    grasp_sub_w_annot.to_csv("test_grasp_sub_w_annot.txt", sep="\t", index=False)
    print("[INFO] save to test_grasp_sub_w_annot.txt")
