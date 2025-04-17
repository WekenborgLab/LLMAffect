######################
##plots_sadness_bias##
######################

#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

#rating of human 1
rater1 = "rater 1"
data_h1 = {
    "n": [1, 2, 3, 4, 5],
    "Rating_sadness": [18, 13, 16, 17, 22],
    "Rating_neutral": [10, 10, 14, 10, 11]
}
#rating of human 2
rater2 = "rater 2"
data_h2 = {
    "n": [1, 2, 3, 4, 5],
    "Rating_sadness": [15, 10, 16, 17, 22],
    "Rating_neutral": [8, 8, 9, 9, 11]
}
#rating of human 3
rater3 = "rater 3"
data_h3 = {
    "n": [1, 2, 3, 4, 5],
    "Rating_sadness": [14, 6, 12, 11, 16],
    "Rating_neutral": [4, 4, 9, 7, 6]
}

##calculate mean and SD
df3 = pd.DataFrame(data_h3)
df2 = pd.DataFrame(data_h2)
df1 = pd.DataFrame(data_h1)

mean_sadness1 = df1["Rating_sadness"].mean()
std_sadness1 = df1["Rating_sadness"].std()

mean_neutral1 = df1["Rating_neutral"].mean()
std_neutral1 = df1["Rating_neutral"].std()
print(f"Means_1: {mean_neutral1}; {mean_sadness1}, SDs_1: {std_neutral1}, {std_sadness1}")

# mean and SDs of differences between neutral and sadness induciton rating 
diff1 = df1['Rating_sadness'] - df1['Rating_neutral']
mean_diff1 = diff1.mean()
std_diff1 = diff1.std(ddof=1)
n1 = len(diff1)
se1 = std_diff1 / np.sqrt(n1)
print(f"")

#calculate mean and SDs
ratings1 = ['Rating_neutral', 'Rating_sadness']
means1 = df1[ratings1].mean()
stds1 = df1[ratings1].std()

#t-Test
t_stat1, p_val1 = ttest_rel(df1['Rating_neutral'], df1['Rating_sadness'])

#Confidence Intervals for paired samples
from scipy.stats import t
ci_low1 = mean_diff1 - t.ppf(0.975, df=n1-1) * se1
ci_high1 = mean_diff1 + t.ppf(0.975, df=n1-1) * se1

#Cohen's d for paired samples
cohen_d1 = mean_diff1 / std_diff1

print(f"Cohens d1: {cohen_d1}, CI_low1: {ci_low1}, CI_high1: {ci_high1}, t-Value1: {t_stat1}, p-value1: {p_val1}")

#asterisk per p-value 
def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

#colors 
colors = ['grey', '#4293C2']

#Bar plots
fig, ax = plt.subplots(figsize=(4, 5))
bars = ax.bar(ratings1, means1, yerr=stds1, capsize=10, color=colors, edgecolor='black')

#scatter-points
for i, col in enumerate(ratings1):
    y = df1[col].values
    x = np.random.normal(i, 0.05, size=len(y))
    #ax.scatter(x, y, color='black', alpha=0.6) #optional

#axis and title
ax.set_title("Negative sentence completion", fontsize=14)
ax.set_ylabel(f"Rating of {rater1}", fontsize=12)
ax.set_xticks(range(len(ratings1)))
ax.set_xticklabels(['Neutral', 'Sadness induction'], fontsize=11)
ax.set_ylim(0, df1[ratings1].values.max() + 5)

#significance line & asterisk
stars = significance_stars(p_val1)
y_max = max(means1 + stds1) + 1
h = 1
x1, x2 = 0, 1
ax.plot([x1, x1, x2, x2], [y_max, y_max + h, y_max + h, y_max], lw=1.5, c='black')
ax.text((x1 + x2) * 0.5, y_max + h + 0.3, stars, ha='center', va='bottom', color='black', fontsize=12)

plt.tight_layout()
plt.savefig(f"Plots/plot_Bias_sadness_{rater1}.jpg", bbox_inches="tight")
plt.savefig(f"Plots_SVG/plot_Bias_sadness_{rater1}.svg", bbox_inches="tight")
plt.show()

mean_sadness2 = df2["Rating_sadness"].mean()
std_sadness2 = df2["Rating_sadness"].std()

mean_neutral2 = df2["Rating_neutral"].mean()
std_neutral2 = df2["Rating_neutral"].std()
print(f"Means_2: {mean_neutral2}; {mean_sadness2}, SDs_2: {std_neutral2}, {std_sadness2}")

#mean and SDs for differences
diff2 = df2['Rating_sadness'] - df2['Rating_neutral']
mean_diff2 = diff2.mean()
std_diff2 = diff2.std(ddof=1)
n2 = len(diff2)
se2 = std_diff2 / np.sqrt(n2)

#calculate means and SDs 
ratings2 = ['Rating_neutral', 'Rating_sadness']
means2 = df2[ratings2].mean()
stds2 = df2[ratings2].std()

#t-Test
t_stat2, p_val2 = ttest_rel(df2['Rating_neutral'], df2['Rating_sadness'])

#confidenceinterval for paired samples 
ci_low2 = mean_diff2 - t.ppf(0.975, df=n2-1) * se2
ci_high2 = mean_diff2 + t.ppf(0.975, df=n2-1) * se2

# Effektgröße: Cohen's d für gepaarte Daten
cohen_d2 = mean_diff2 / std_diff2

print(f"Cohens d2: {cohen_d2}, CI_low2: {ci_low2}, CI_high2: {ci_high2}, t-Value2: {t_stat2}, p-value2: {p_val2}")

#asterisk per p-value
def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

#olors
colors = ['grey', '#4293C2']

#Bar plots
fig, ax = plt.subplots(figsize=(4, 5))
bars = ax.bar(ratings2, means2, yerr=stds2, capsize=10, color=colors, edgecolor='black')

#scatter points (optional)
for i, col in enumerate(ratings2):
    y = df2[col].values
    x = np.random.normal(i, 0.05, size=len(y))  
    #ax.scatter(x, y, color='black', alpha=0.6) #optional

#Axis & title
ax.set_title("Negative sentence completion", fontsize=14)
ax.set_ylabel(f"Rating of {rater2}", fontsize=12)
ax.set_xticks(range(len(ratings2)))
ax.set_xticklabels(['Neutral', 'Sadness induction'], fontsize=11)
ax.set_ylim(0, df2[ratings2].values.max() + 5)

#significance line and asterisk 
stars = significance_stars(p_val2)
y_max = max(means2 + stds2) + 1
h = 1
x1, x2 = 0, 1
ax.plot([x1, x1, x2, x2], [y_max, y_max + h, y_max + h, y_max], lw=1.5, c='black')
ax.text((x1 + x2) * 0.5, y_max + h + 0.3, stars, ha='center', va='bottom', color='black', fontsize=12)

plt.tight_layout()
plt.savefig(f"Plots/plot_Bias_sadness_{rater2}.jpg", bbox_inches="tight")
plt.savefig(f"Plots_SVG/plot_Bias_sadness_{rater2}.svg", bbox_inches="tight")
plt.show()

mean_sadness3 = df3["Rating_sadness"].mean()
std_sadness3 = df3["Rating_sadness"].std()

mean_neutral3 = df3["Rating_neutral"].mean()
std_neutral3 = df3["Rating_neutral"].std()

print(f"Means_3: {mean_neutral3}; {mean_sadness3}, SDs_1: {std_neutral3}, {std_sadness3}")

#mean and SDs of differences
diff3 = df3['Rating_sadness'] - df3['Rating_neutral']
mean_diff3 = diff3.mean()
std_diff3 = diff3.std(ddof=1)
n3 = len(diff3)
se3 = std_diff3 / np.sqrt(n3)

#calculate mean and SDs 
ratings3 = ['Rating_neutral', 'Rating_sadness']
means3 = df3[ratings3].mean()
stds3 = df3[ratings3].std()

#t-Test
t_stat3, p_val3 = ttest_rel(df3['Rating_neutral'], df3['Rating_sadness'])

#confidence interval for paired samples
from scipy.stats import t
ci_low3 = mean_diff3 - t.ppf(0.975, df=n3-1) * se3
ci_high3 = mean_diff3 + t.ppf(0.975, df=n3-1) * se3

#Cohen's d for paired samples
cohen_d3 = mean_diff3 / std_diff3

print(f"Cohens d3: {cohen_d3}, CI_low3: {ci_low3}, CI_high3: {ci_high3}, t-Value3: {t_stat3}, p-value3: {p_val3}")

#Asterisk per p-Value
def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

#color
colors = ['grey', '#4293C2']

#Bar Plot
fig, ax = plt.subplots(figsize=(4, 5))
bars = ax.bar(ratings3, means3, yerr=stds3, capsize=10, color=colors, edgecolor='black')

#Scatter points (optional)
for i, col in enumerate(ratings3):
    y = df3[col].values
    x = np.random.normal(i, 0.05, size=len(y)) 
    #ax.scatter(x, y, color='black', alpha=0.6) #optional

#Axis and title
ax.set_title("Negative sentence completion", fontsize=14)
ax.set_ylabel(f"Rating of {rater3}", fontsize=12)
ax.set_xticks(range(len(ratings3)))
ax.set_xticklabels(['Neutral', 'Sadness induction'], fontsize=11)
ax.set_ylim(0, df3[ratings3].values.max() + 5)

#significance line and asterisk
stars = significance_stars(p_val3)
y_max = max(means3 + stds3) + 1
h = 1
x1, x2 = 0, 1
ax.plot([x1, x1, x2, x2], [y_max, y_max + h, y_max + h, y_max], lw=1.5, c='black')
ax.text((x1 + x2) * 0.5, y_max + h + 0.3, stars, ha='center', va='bottom', color='black', fontsize=12)

plt.tight_layout()
plt.savefig(f"Plots/plot_Bias_sadness_{rater3}.jpg", bbox_inches="tight")
plt.savefig(f"Plots_SVG/plot_Bias_sadness_{rater3}.svg", bbox_inches="tight")
plt.show()

#put all in one Data Frame
df1['rater'] = 'rater1'
df2['rater'] = 'rater2'
df3['rater'] = 'rater3'
df_all = pd.concat([df1, df2, df3], ignore_index=True)

#calculate differences
diff = df_all["Rating_sadness"] - df_all["Rating_neutral"]
mean_diff = diff.mean()
std_diff = diff.std(ddof=1)
n = len(diff)
se = std_diff / np.sqrt(n)

#t-Test
t_stat, p_val = ttest_rel(df_all["Rating_sadness"], df_all["Rating_neutral"])

#95%-CI
ci_low = mean_diff - t.ppf(0.975, df=n-1) * se
ci_high = mean_diff + t.ppf(0.975, df=n-1) * se

#cohen's d
cohen_d = mean_diff / std_diff

#output
print(f"t_all({n-1}) = {t_stat}, p_all = {p_val:.3f}, 95% CI_all [{ci_low}, {ci_high}], d_all = {cohen_d:.2f}")

ratings = ['Rating_neutral', 'Rating_sadness']
means = df_all[ratings].mean()
stds = df_all[ratings].std()
print(f"Means_alle: {means}, SDs_alle: {stds}")
colors = ['grey', '#4293C2']

fig, ax = plt.subplots(figsize=(4, 5))
bars = ax.bar(ratings, means, yerr=stds, capsize=10, color=colors, edgecolor='black')

#significance lines and asteriks for all plots
stars = significance_stars(p_val)
y_max = max(means + stds) + 1 
h = 1 
x1, x2 = 0, 1
ax.plot([x1, x1, x2, x2], [y_max, y_max + h, y_max + h, y_max], lw=1.5, c='black')
ax.text((x1 + x2) * 0.5, y_max + h + 0.3, stars, ha='center', va='bottom', color='black', fontsize=12)

#description
ax.set_title("Combined rating across raters", fontsize=14)
ax.set_ylabel("Rating", fontsize=12)
ax.set_xticklabels(['Neutral', 'Sadness induction'], fontsize=11)
ax.set_ylim(0, df_all[ratings].values.max() + 5)

plt.tight_layout()
plt.savefig(f"Plots/plot_Bias_sadness_all.jpg", bbox_inches="tight")
plt.savefig(f"Plots_SVG/plot_Bias_sadness_all.svg", bbox_inches="tight")
plt.show()

