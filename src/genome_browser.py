# UCSC Genome Browser
import os
import sys
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
from itertools import repeat
import wget
import ast
import multiprocessing as mp
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.pool import NullPool


_db_url = {
    "drivername": 'mysql+pymysql',
    "host": "genome-mysql.cse.ucsc.edu",
    "port": "3306",
    "username": "genome",
    "password": "",
    "database": 'hg19',
    "query": {'charset': 'utf8'}
}

_seq_url = "ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz"
_chrom_set = ["chr"+str(i) for i in range(1, 23)] + ["chrX", "chrY"]

def fetch_seq(df, df_total, chrom, coord_version, window_size=1000):

    print("[INFO] Sequencing fetch ref+alt+haplotype+2strands alleles of {} of length {} ......".format(chrom, window_size))
    df['seq_ref_1'] = ''
    df['seq_ref_2'] = ''
    df['seq_alt_1'] = ''
    df['seq_alt_2'] = ''
    df['seq_hap_1'] = ''
    df['seq_hap_2'] = ''
    n_empty = 0

    if coord_version == 'hg19':
        dna_chr = list(SeqIO.parse("chromFa_hg19/{}.fa".format(chrom), "fasta"))[0].seq
    elif coord_version == 'hg38':
        dna_chr = list(SeqIO.parse("chromFa_hg38/{}.fa".format(chrom), "fasta"))[0].seq

    for ind, row in tqdm(df.iterrows()):
        start = row['pos'] - window_size // 2
        end = row['pos'] + window_size // 2

        nearby = df_total.loc[(df_total['pos'] >= start) & (df_total['pos'] < end)]

        if start >= 0 and end <= len(dna_chr):
            ref_seq = dna_chr[start: end]
            alt_seq = dna_chr[start: row['pos']-1] + row['alt'] + dna_chr[row['pos']: end]
            df.ix[ind, 'seq_ref_1'] = ref_seq
            df.ix[ind, 'seq_ref_2'] = ref_seq.reverse_complement()
            df.ix[ind, 'seq_alt_1'] = alt_seq
            df.ix[ind, 'seq_alt_2'] = alt_seq.reverse_complement()
            hap_seq = list(ref_seq)
            for i, v in nearby.iterrows():
                hap_seq[v['pos']-1-start] = v['alt']
            hap_seq = Seq(''.join(hap_seq))
            df.ix[ind, 'seq_hap_1'] = hap_seq
            df.ix[ind, 'seq_hap_2'] = hap_seq.reverse_complement()

        else:
            n_empty += 1

    df = df.dropna(subset=['seq_ref_1', 'seq_ref_2', 'seq_alt_1', 'seq_alt_2', 'seq_hap_1', 'seq_hap_2'])
    print('[INFO] n_empty of {} is: {}'.format(chrom, n_empty))
    return df

def fast_fetch_seq(df, chrom, coord_version, window_size=1000):
    cores = mp.cpu_count()
    pool = mp.Pool(cores)
    df_list = np.array_split(df, cores)
    df_seq = pd.concat(pool.starmap(fetch_seq, zip(df_list, repeat(df[['pos', 'alt']]), repeat(chrom), repeat(coord_version), repeat(window_size))))
    pool.close()
    pool.join()

    return df_seq

def fetch_metadata(rsid):

    db = create_engine(URL(**_db_url), poolclass=NullPool)
    db.execute("SET sql_mode = 'NO_UNSIGNED_SUBTRACTION'")

    snps = ", ".join("'" + x + "'" for x in rsid)

    query = '''
            SELECT
                s.name, s.chrom, s.chromStart, s.chromEnd
            FROM
                snp146 s
            WHERE
                s.name IN  ( ''' + snps + ''')
            '''

    rows = db.execute(query)

    metadata = pd.DataFrame(rows.fetchall())
    metadata.columns = rows.keys()
    metadata = metadata.rename(columns={"name":"rsid"})

    return metadata

def fast_fetch_metadata(rsid, save=None):
    # parallel metadata query
    cores = mp.cpu_count()
    pool = mp.Pool(cores)
    rsid_split = np.array_split(rsid, cores)
    metadata = pd.concat(pool.map(fetch_metadata, rsid_split))
    pool.close()
    pool.join()

    metadata = metadata.drop_duplicates(subset="rsid", keep="first").reset_index(drop=True)
    metadata = metadata.dropna(axis=0, how="any").reset_index(drop=True)
    # remove the cases from chromosome "chrUn_gl000248" etc
    metadata = metadata[metadata["chrom"].isin(_chrom_set)]

    if save is not None:
        metadata.to_csv(save, sep="\t", index=False)
        print("[INFO] metadata generated: ", save, "!")

    return metadata

if __name__ == '__main__':
    df = pd.DataFrame({'pos':[50000000, 15000000, 32000000, 20000000, 90900999, 50000010], 'chr': ['chr13', 'chr13', 'chr13', 'chr13', 'chr13', 'chr13'], 'alt':['1', '2', '3', '4', '5', '6']})
    seqdata = fast_fetch_seq(df, 'chr13', 'hg38', 1000)
    for k in range(len(seqdata['seq_hap_1'])):
        for i in range(len(seqdata['seq_alt_1'][k])):
            if seqdata['seq_alt_1'][k][i] != seqdata['seq_alt_1'][k][i]:
                print(i, seqdata['seq_alt_1'][k][i], seqdata['seq_alt_1'][k][i])
    print(seqdata.ix[0, 'seq_ref_1'])
    print(seqdata.ix[0, 'seq_ref_2'])
