import positive
import pandas as pd

# generate positive dataset from annotated grasp database
window_size = 1000000
positive_ob = positive.positive("grasp_sub_w_annot.txt")
grasp_sub_w_annot_positive = positive_ob.fast_generate_positive(window_size)
grasp_sub_w_annot_positive = grasp_sub_w_annot_positive.sort_values(["ID"]).reset_index(drop=True)
grasp_sub_w_annot_positive.to_csv("grasp_sub_w_annot_positive.txt", sep="\t", index=False)
print("[INFO] save to grasp_sub_w_annot_positive.txt")
positive_sub = grasp_sub_w_annot_positive[grasp_sub_w_annot_positive["positive"]==True]
positive_sub = positive_sub.sort_values(["ID"]).reset_index(drop=True)
positive_sub['ProxyFrom'] = positive_sub['SNPid(dbSNP134)'] # add SNAP PROXY SNP for locus sampling and avgrank
positive_sub.to_csv("positive_sub.txt", sep="\t", index=False)

# merge sub with full
grasp_full = pd.read_csv("grasp_full.txt", sep="\t")
print("[INFO] load in grasp_full.txt!")
grasp_full_w_annot_positive = pd.merge(grasp_full, grasp_sub_w_annot_positive[["ID", "annotation", "positive"]], on="ID")
grasp_full_w_annot_positive = grasp_full_w_annot_positive.sort_values(["ID"]).reset_index(drop=True)
grasp_full_w_annot_positive.to_csv("grasp_full_w_annot_positive.txt", sep="\t", index=False)
print("[INFO] save to grasp_full_w_annot_positive.txt")
positive_full = grasp_full_w_annot_positive[grasp_full_w_annot_positive["positive"]==True]
positive_full = positive_full.sort_values(["ID"]).reset_index(drop=True)
positive_full['ProxyFrom'] = positive_full['SNPid(dbSNP134)'] # add SNAP PROXY SNP for locus sampling and avgrank
positive_full.to_csv("positive_full.txt", sep="\t", index=False)
print("[INFO] save to positive_full.txt")
