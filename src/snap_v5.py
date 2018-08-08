import sys
import math
import time
import http
import pickle
import itertools
import numpy as np
import pandas as pd
import multiprocessing as mp
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from io import StringIO, BytesIO
from tqdm import tqdm

def _query(rsid_list, p, search_kwargs, verbose=False):
    time0 = time.time()

    fail_list = []
    partial_list = []

    #url = 'http://archive.broadinstitute.org/mammals/haploreg/haploreg.php'
    url = 'https://pubs.broadinstitute.org/mammals/haploreg/haploreg.php'

    param = {
        'query': ','.join(rsid_list),
        'ldThresh': search_kwargs['ldThresh'], # between [0.0, 1.0]
        'ldPop': p, # 'AFR', 'AMR', 'ASN', 'EUR'
        'epi': search_kwargs['epi'], # 'vanilla', 'imputed', 'methyl', 'acetyl'
        'cons': search_kwargs['cons'], # 'gerp', 'siphy', 'both'
        'genetypes': search_kwargs['genetypes'], # 'gencode', 'refseq', 'both'
        'trunc': search_kwargs['trunc'], # 0, 1, 2, 3, 4, 5, 1000
        'oligo': search_kwargs['oligo'], # 1, 6, 1000
        'output': search_kwargs['output'], # 'html', 'text'
    }

    data = urlencode(param)
    data = data.encode('ascii')
    req = Request(url, data)

    response = urlopen(req)

    try:
        page = response.read()
    except (http.client.IncompleteRead) as e:
        print('[WARNING] partial read')
        page = e.partial
        partial_list.extend(rsid_list)
    except:
        print('[ERROR] snap page read fails!')
        fail_list.extend(rsid_list)
        print(fail_list)
        return None, partial_list, fail_list

    response.close()

    dfm = pd.read_csv(BytesIO(page), encoding='utf-8', header=0, sep='\t')

    # reformat
    if dfm.shape[0] > 0:
        dfm['chr'] = dfm.apply(lambda x: 'chr' + str(x['chr']), axis=1)
        dfm['pop'] = p
    else: # when any query snp does not exist in 1000Genome Phase I data
        fail_list.extend(rsid_list)
        print(fail_list)
        return None, partial_list, fail_list

    time1 = time.time()

    if verbose:
        print('[INFO] [#input={num_input}, #output={num_output}], population={population}, '
        'ld_thresh={ld_thresh}, epigenomes_source={epi}, conservation_score={cons}, '
        'position_to_genetypes={genetypes}, time={time:.2f}'.format(num_input=len(rsid_list),
                                                   num_output=dfm.shape[0],
                                                   population=p,
                                                   ld_thresh=search_kwargs['ldThresh'],
                                                   epi=search_kwargs['epi'],
                                                   cons=search_kwargs['cons'],
                                                   genetypes=search_kwargs['genetypes'],
                                                   time=time1-time0))

    return dfm, partial_list, fail_list

def query(rsid_list, search_kwargs, verbose=False):
    dfm = []
    partial_list = []
    fail_list = []

    __chrom_set = ["chr"+str(i) for i in range(1, 23)] + ["chrX", "chrY"]

    for p in search_kwargs['ldPop']:
        dfm_p, pl, fl = _query(rsid_list, p, search_kwargs, verbose)
        if dfm_p is not None:
            dfm.append(dfm_p)
        partial_list.extend(pl)
        fail_list.extend(fl)

    if len(dfm) > 0:
        dfm = pd.concat(dfm).sort_values(['rsID', 'query_snp_rsid', 'pop'], ascending=[True, True, True])

        # variant without chrom info or rsid are excluded
        dfm = dfm.loc[dfm['chr'].isin(__chrom_set)]

        # variants without allele frequencies are excluded
        dfm = dfm.dropna(subset=['AFR', 'AMR', 'ASN', 'EUR'])

        # change datetypes
        dfm.AFR = dfm.AFR.astype(float)
        dfm.AMR = dfm.AMR.astype(float)
        dfm.ASN = dfm.ASN.astype(float)
        dfm.EUR = dfm.EUR.astype(float)
        dfm.pos_hg38 = dfm.pos_hg38.astype(int)
        # indels are excluded
        dfm = dfm.loc[(dfm['ref'].str.len()==1) & (dfm['alt'].str.len()==1)]

        # SNPs without rsID are excluded
        dfm = dfm.loc[dfm['rsID'].str.contains('rs')]

        # rare variants (mutations) are excluded
        dfm = dfm.loc[(dfm['AFR'] >= search_kwargs['maf_thresh']) |
                      (dfm['AMR'] >= search_kwargs['maf_thresh']) |
                      (dfm['ASN'] >= search_kwargs['maf_thresh']) |
                      (dfm['EUR'] >= search_kwargs['maf_thresh'])]

        # if query snp removed, all its proxy snps are excluded
        rsid_list_missing = list(set(rsid_list) - set(dfm.rsID))
        if len(rsid_list_missing) > 0:
            dfm = dfm.loc[~dfm['query_snp_rsid'].isin(rsid_list_missing)]

        partial_list = list(set(partial_list))
        fail_list = list(set(fail_list))

        return dfm, partial_list, fail_list

    else:
        partial_list = list(set(partial_list))
        fail_list = list(set(fail_list))
        return None, partial_list, fail_list

def fast_get_locus_map(rsid_list, search_kwargs, verbose=False):
    num_cores = mp.cpu_count()
    chunk_size_limit = 10000

    locus_map_total = None
    attempt = 0
    max_attempt = search_kwargs['max_attempt']

    while attempt < max_attempt:
        attempt += 1
        chunk_size = math.ceil(len(rsid_list) / num_cores) if math.ceil(len(rsid_list) / num_cores) <= chunk_size_limit else chunk_size_limit
        rsid_split = [rsid_list[i:i+chunk_size] for i in range(0, len(rsid_list), chunk_size)]
        args = zip(rsid_split, [search_kwargs]*len(rsid_split), [verbose]*len(rsid_split))

        pool = mp.Pool(num_cores)
        results = pool.starmap(query, args)
        pool.close()
        pool.join()
        locus_map = pd.concat([res[0] for res in results])
        locus_map_total = pd.concat([locus_map_total, locus_map]).drop_duplicates(subset=['rsID', 'query_snp_rsid', 'pop'], keep='first')
        partial_list = list(itertools.chain(*[res[1] for res in results]))
        fail_list = list(itertools.chain(*[res[2] for res in results]))
        print('# locus_map_total: {}'.format(locus_map_total.shape[0]))

    return locus_map_total, partial_list, fail_list

if __name__ == '__main__':
    start_time = time.time()

    rsid_list = ['rs3', 'rs4', 'rs123', 'rs321']

    search_kwargs = {}
    search_kwargs['ldThresh'] = 0.80
    search_kwargs['ldPop'] = ['AFR', 'AMR', 'ASN', 'EUR']
    search_kwargs['epi'] = 'vanilla'
    search_kwargs['cons'] = 'both'
    search_kwargs['genetypes'] = 'both'
    search_kwargs['trunc'] = 1000
    search_kwargs['oligo'] = 1000
    search_kwargs['output'] = 'text'
    search_kwargs['maf_thresh'] = 0.05
    search_kwargs['max_attempt'] = 1

    locus_map, partial_list, fail_list = fast_get_locus_map(rsid_list, search_kwargs, True)
    locus_map.to_csv('tmp_1.txt', sep='\t', index=False)
    print('[INFO] tmp_1.txt saved!')

    with open('partial_list.txt', 'w') as pl:
        for rsid in partial_list:
            pl.write('{}\n'.format(rsid))
    print('[INFO] partial_list.txt saved!')

    with open('fail_list.txt', 'w') as fl:
        for rsid in fail_list:
            fl.write('{}\n'.format(rsid))
    print('[INFO] fail_list.txt saved!')

    print(time.time() - start_time)
