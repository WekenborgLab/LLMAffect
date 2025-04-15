##############################
##Mood Bias Data Calculation##
##############################

#import package
import pandas as pd


models = ["gpt", "scout", "qwen", "8B_unquantisiert", "70B_unquantisiert", "maverick"]

for model in models:
    #get Data Frame
    df = pd.read_csv(f"results_{model}.csv")

    #means and SDs for Sadness 
    ##get right VAS
    vas_scales = ["vas_scales_sadness"]

    #group and calculate means and CIs for VAS
    df_grouped = df.groupby(["promptid-trialid", "section_number"])[vas_scales].agg(["mean", "std", "count"]).reset_index()
    print(f"DF_Grouped: {df_grouped}")

    #save as Excel Sheet
    df_grouped.to_excel("Excle_Data_summary_sadness.xlsx")
    print("Saved to Excel")

