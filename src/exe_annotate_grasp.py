import annotate
import pandas as pd

# annotate SNPs in GRASP database
grasp_sub = pd.read_csv("grasp_sub.txt", sep="\t", converters={"PMID": str, "NHLBIkey": str}).rename(columns={'chr(hg19)': 'chr', 'pos(hg19)': 'pos'})
annotate_ob = annotate.annotate(grasp_sub, 'hg19')
grasp_sub_w_annot = annotate_ob.fast_annotate_snp_list()
grasp_sub_w_annot = grasp_sub_w_annot.sort_values(["ID"]).reset_index(drop=True)
grasp_sub_w_annot.to_csv("grasp_sub_w_annot.txt", sep="\t", index=False)
print("[INFO] save to grasp_sub_w_annot.txt")
