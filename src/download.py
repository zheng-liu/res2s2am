import os
import wget
import pandas as pd

__db_url = {
    "drivername": 'mysql+pymysql',
    "host": "genome-mysql.cse.ucsc.edu",
    "port": "3306",
    "username": "genome",
    "password": "",
    "database": 'hg19',
    "query": {'charset': 'utf8'}
}
__chrom_set = ["chr"+str(i) for i in range(1, 23)] + ["chrX", "chrY"]
__pop_set = ['ASW', 'CEU', 'CHB', 'CHD', 'GIH', 'JPT', 'LWK', 'MEX', 'MKK', 'TSI', 'YRI']

__grasp_url = "https://s3.amazonaws.com/NHLBI_Public/GRASP/GraspFullDataset2.zip"
__hg19_seq_url = "ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz"
__hg38_seq_url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.chromFa.tar.gz"
__1kg_url = 'ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/'
__hapmap_url = 'ftp://ftp.ncbi.nlm.nih.gov/hapmap/ld_data/2009-04_rel27/'
__phantomjs_url = 'https://bitbucket.org/ariya/phantomjs/downloads/phantomjs-2.1.1-linux-x86_64.tar.bz2'
__regulomedb_url = 'http://www.regulomedb.org/downloads/'

def seq_db_download():
    # os.system("rm chromFa.tar.gz")
    print("[INFO] downloading hg19 chromFa.tar.gz from UCSC")
    wget.download(__hg19_seq_url)
    print("\n[INFO] chromFa.tar.gz downloaded.")
    os.system("rm -rf chromFa_hg19")
    os.system("mkdir chromFa_hg19")
    os.system("tar xvzf chromFa.tar.gz -C ./chromFa_hg19/")
    print("[INFO] hg19 chromFa.tar.gz unzipped into ./chromFa directory")

    print("[INFO] downloading hg38 chromFa.tar.gz from UCSC")
    wget.download(__hg38_seq_url)
    print("\n[INFO] hg38 chromFa.tar.gz downloaded.")
    os.system("rm -rf chromFa_hg38")
    os.system("mkdir chromFa_hg38")
    os.system("tar xvzf hg38.chromFa.tar.gz -C ./chromFa_hg38/")
    os.system('mv ./chromFa_hg38/chroms/* ./chromFa_hg38/')
    os.system('rm -rf ./chromFa_hg38/chroms')
    print("[INFO] hg38 chromFa.tar.gz unzipped into ./chromFa_hg38 directory")

def download_grasp():

    if not os.path.exists("GRASP2fullDataset"):
        wget.download(__grasp_url)
        print("\n[INFO] GraspFullDataset2.zip downloaded")
        os.system("unzip GraspFullDataset2.zip")
        print("[INFO] Unzipped to GRASP2fullDataset")
    # else:
    #     print("[INFO] loading the local GRASP2fullDataset")

    grasp_full_raw = pd.read_csv("GRASP2fullDataset", sep="\t", encoding="ISO-8859-1", \
        converters={"NHLBIkey":str, "PMID": str, "InMiRNA": str, "InMiRNABS":str, \
        "dbSNPClinStatus":str, "ConservPredTFBS":str, "HumanEnhancer": str, \
        "RNAedit": str, "PolyPhen2":str, "LS-SNP":str, "UniProt":str})
    print("[INFO] load in GRASP2fullDataset!")
    grasp_full_raw["SNPid(dbSNP134)"] = "rs" + grasp_full_raw["SNPid(dbSNP134)"].astype(str)
    grasp_full_raw["chr(hg19)"] = "chr" + grasp_full_raw["chr(hg19)"].astype(str)
    grasp_full_raw["pos(hg19)"] = grasp_full_raw["pos(hg19)"].astype("int64")

    grasp_full_raw["NHLBIkey"] = grasp_full_raw["NHLBIkey"].astype(str)
    print("[INFO] NHLBIkey convert to string")
    grasp_full_raw["PMID"] = grasp_full_raw["PMID"].astype(str)
    print("[INFO] PMID convert to string")
    grasp_full_raw["InMiRNA"] = grasp_full_raw["InMiRNA"].astype(str)
    print("[INFO] InMiRNA convert to string")
    grasp_full_raw["InMiRNABS"] = grasp_full_raw["InMiRNABS"].astype(str)
    print("[INFO] InMiRNABS convert to string")
    grasp_full_raw["dbSNPClinStatus"] = grasp_full_raw["dbSNPClinStatus"].astype(str)
    print("[INFO] dbSNPClinStatus convert to string")
    grasp_full_raw["ConservPredTFBS"] = grasp_full_raw["ConservPredTFBS"].astype(str)
    print("[INFO] ConservPredTFBS convert to string")
    grasp_full_raw["HumanEnhancer"] = grasp_full_raw["HumanEnhancer"].astype(str)
    print("[INFO] HumanEnhancer convert to string")
    grasp_full_raw["RNAedit"] = grasp_full_raw["RNAedit"].astype(str)
    print("[INFO] RNAedit convert to string")
    grasp_full_raw["PolyPhen2"] = grasp_full_raw["PolyPhen2"].astype(str)
    print("[INFO] PolyPhen2 convert to string")
    grasp_full_raw["LS-SNP"] = grasp_full_raw["LS-SNP"].astype(str)
    print("[INFO] LS-SNP convert to string")
    grasp_full_raw["UniProt"] = grasp_full_raw["UniProt"].astype(str)
    print("[INFO] UniProt convert to string")

    grasp_full_raw["ID"] = grasp_full_raw.index
    grasp_sub_raw = grasp_full_raw[["ID", "NHLBIkey", "PMID", "SNPid(dbSNP134)", "chr(hg19)", "pos(hg19)", "Phenotype", "Pvalue"]]
    grasp_full_raw.to_csv("grasp_full.txt", sep="\t", index=False)
    print("[INFO] save to grasp_full.txt")
    grasp_sub_raw.to_csv("grasp_sub.txt", sep="\t", index=False)
    print("[INFO] save to grasp_sub.txt")

def download_onekg():
    if not os.path.exists('onekg'):
        os.system('mkdir onekg')
        for chrom in __chrom_set:
            if chrom is 'chrX':
                url1 = __1kg_url + '/ALL.{}.phase3_shapeit2_mvncall_integrated_v1b.20130502.genotypes.vcf.gz'.format(chrom)
                url2 = __1kg_url + '/ALL.{}.phase3_shapeit2_mvncall_integrated_v1b.20130502.genotypes.vcf.gz.tbi'.format(chrom)
            elif chrom is 'chrY':
                url1 = __1kg_url + '/ALL.{}.phase3_integrated_v2a.20130502.genotypes.vcf.gz'.format(chrom)
                url2 = __1kg_url + '/ALL.{}.phase3_integrated_v2a.20130502.genotypes.vcf.gz.tbi'.format(chrom)
            else:
                url1 = __1kg_url + '/ALL.{}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz'.format(chrom)
                url2 = __1kg_url + '/ALL.{}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz.tbi'.format(chrom)
            wget.download(url1, './onekg')
            wget.download(url2, './onekg')
            print('\n[INFO] {} finished downloading!'.format(chrom))

def download_hapmap():
    if not os.path.exists('hapmap'):
        os.system('mkdir hapmap')
        for chrom in __chrom_set[:-1]:
            for pop in __pop_set:
                url = __hapmap_url + 'ld_{chrom}_{pop}.txt.gz'.format(chrom=chrom, pop=pop)
                wget.download(url, './hapmap')
            print('\n[INFO] downloaded {} hapmap!'.format(chrom))

        for chrom in __chrom_set[:-1]:
            for pop in __pop_set:
                hapmap_file = './hapmap/ld_{chrom}_{pop}.txt.gz'.format(chrom=chrom, pop=pop)
                os.system('gunzip {}'.format(hapmap_file))
            print('[INFO] unzipped {} hapmap!'.format(chrom))

def download_regulomedb():
    if not os.path.exists('regulomedb'):
        os.system('mkdir regulomedb')
        for i in range(1, 8):
            url = __regulomedb_url + 'RegulomeDB.dbSNP132.Category{}.txt.gz'.format(i)
            wget.download(url, './regulomedb')
            print('\n[INFO] downloaded RegulomeDB {} category!'.format(i))

    for i in range(1, 8):
        regulomedb_file = './regulomedb/RegulomeDB.dbSNP132.Category{}.txt.gz'.format(i)
        os.system('gunzip {}'.format(regulomedb_file))
        print('[INFO] unzipped {} category!'.format(i))


def download_phantomjs():
    if not os.path.exists("phantomjs-2.1.1-linux-x86_64.tar.bz2"):
        wget.download(__phantomjs_url)
        print("\n[INFO] phantomjs-2.1.1-linux-x86_64.tar.bz2 downloaded")
        os.system("tar xvjf phantomjs-2.1.1-linux-x86_64.tar.bz2")
        print("[INFO] Unzipped to GRASP2fullDataset")
