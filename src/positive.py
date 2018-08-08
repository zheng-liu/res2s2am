import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

class positive():
    __chrom_set = ["chr"+str(i) for i in range(1, 23)] + ["chrX", "chrY"]

    def __init__(self, filename="grasp_sub_w_annot.txt"):
        self.cores = mp.cpu_count()
        self.grasp_w_annotation = pd.read_csv(filename, sep="\t")

    def generate_positive(self, grasp_groups, window_size):

        for name, group in tqdm(grasp_groups, total=len(grasp_groups)):
            group["positive"] = None

            # protein-coding SNPs are NOT positive
            group.loc[group["annotation"]=="pcexon", "positive"] = False

            df_pcexon = group[(group["annotation"]=="pcexon") & (group["Pvalue"] < 5e-8)] # associated pcexon
            for chromStart_pcexon, pmid, phenotype in zip(df_pcexon["pos"], df_pcexon["PMID"], df_pcexon["Phenotype"]):
                group.loc[(group["pos"] >= chromStart_pcexon - window_size/2) & \
                    (group["pos"] <= chromStart_pcexon + window_size/2) & \
                    (group["PMID"] == pmid) & \
                    (group["Phenotype"] == phenotype), "positive"] = False

            for ind, row in group.iterrows():
                if row["positive"] == True:
                    continue
                if row["positive"] == False:
                    continue
                id_locus = group.loc[(group["pos"] >= row["pos"] - window_size/2) & \
                    (group["pos"] <= row["pos"] + window_size/2) & \
                    (group["annotation"] != "pcexon")]["ID"]
                min_pvalue = group.loc[group["ID"].isin(id_locus), "Pvalue"].min()
                id_min_pvalue = group.loc[group["Pvalue"]==min_pvalue, "ID"]
                group.loc[group["ID"].isin(id_locus), "positive"] = False
                group.loc[(group["ID"].isin(id_min_pvalue)) & \
                    (group["annotation"]!="pcexon"), "positive"] = True

            self.positive_list.append(group)

    # parallel computing of positive cases
    def fast_generate_positive(self, window_size):
        self.manager = mp.Manager()
        self.positive_list = self.manager.list()
        processes = []

        groups = self.grasp_w_annotation.groupby(["chr", "PMID", "Phenotype"])

        grasp_groups_list = np.array_split(groups, self.cores)

        for grasp_groups in grasp_groups_list:
            p = mp.Process(target=self.generate_positive, args=(grasp_groups, window_size))
            processes.append(p)
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        self.grasp_sub_w_annot_positive = pd.concat(self.positive_list)
        self.grasp_sub_w_annot_positive.sort_values(by=['SNPid(dbSNP134)', 'ID', 'positive'], ascending=[True, True, False], inplace=True) # order according to [rsid, positive]
        self.grasp_sub_w_annot_positive.drop_duplicates(subset='SNPid(dbSNP134)', keep='first', inplace=True) # keep all the TRUE if there is a TRUE (the first entry for each SNP rsid)
        self.grasp_sub_w_annot_positive.reset_index(drop=True, inplace=True)
        print("[INFO] generate grasp_sub_w_annot_positive!")

        return self.grasp_sub_w_annot_positive
