##################
##Plots for TSST##
##################

#libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t

#model = "Llama70B"
#model = "GPT4o"
#model = "Scout"
#model = "Qwen"
#model = "maverick"
model = "8b"
# Sample Data: Create a DataFrame with the given values
n = 5
data_maverick = { #maverick
    'prompt_stage': ['Baseline', 'Instruction', 'Preparation', 'Interview', 'Arithmetic Task', 'Debriefing'],
    'mean': [22, 44, 38, 36, 42, 57],  # Means
    'SD': [4.47, 8.94, 27.75, 7.1, 15.17, 32.48],  # Standard deviations
}

data_Qwen = { #Qwen
    'prompt_stage': ['Baseline', 'Instruction', 'Preparation', 'Interview', 'Arithmetic Task', 'Debriefing'],
    'mean': [25, 36, 44, 39, 33, 26],  # Means
    'SD': [7.1, 11.4, 11.4, 5, 6.71, 5.48],  # Standard deviations
}
data_Scout = { #Scout
    'prompt_stage': ['Baseline', 'Instruction', 'Preparation', 'Interview', 'Arithmetic Task', 'Debriefing'],
    'mean': [26, 40, 55, 36, 30, 40],  # Means
    'SD': [13.42, 14.14, 10, 8.37, 8.37, 11.4],  # Standard deviations
}
data_70B = { #70B
    'prompt_stage': ['Baseline', 'Instruction', 'Preparation', 'Interview', 'Arithmetic Task', 'Debriefing'],
    'mean': [25, 36, 46, 32, 34, 32],  # Means
    'SD': [5, 16.73, 15.17, 17.89, 20.74, 19.24],  # Standard deviations
}
data_GPT4o = { #GPT4o
    'prompt_stage': ['Baseline', 'Instruction', 'Preparation', 'Interview', 'Arithmetic Task', 'Debriefing'],
    'mean': [20, 61, 70, 53, 54, 26],  # Means
    'SD': [0, 10.25, 7.9, 6.7, 11.94, 8.94],  # Standard deviations
}
data_8B = { #8B
    'prompt_stage': ['Baseline', 'Instruction', 'Preparation', 'Interview', 'Arithmetic Task', 'Debriefing'],
    'mean': [20, 20, 20, 20, 20, 40],  # Means
    'SD': [0, 0, 0, 0, 0, 24.5],  # Standard deviations
}


#convert to DataFrame
mood_df = pd.DataFrame(data)

#mood/VAS is always stress as these are the plots for the TSST
mood = "stress"
scale = "vas_scales_stress"

#define colors for the mood
mood_colors = {"stress": "#7f7f7f"}

#calculate the lower and upper confidence intervals (CI), t-distribution as n = 5
n = 5
t_val = t.ppf(0.975, df=n-1)
mood_df[f"{scale}_lower_ci"] = mood_df['mean'] - 1.96 * (mood_df['SD'] / (n ** 0.5))
mood_df[f"{scale}_upper_ci"] = mood_df['mean'] + 1.96 * (mood_df['SD'] / (n ** 0.5))

#pointplot for means
sns.pointplot(
    data=mood_df,
    x="prompt_stage",
    y="mean",
    color=mood_colors.get(mood, "#7f7f7f"),
    errorbar=None,
    markers="o",
    linestyles="-"
)

#add CI range manually
plt.fill_between(
    mood_df["prompt_stage"],
    mood_df[f"{scale}_lower_ci"],
    mood_df[f"{scale}_upper_ci"],  
    color=mood_colors.get(mood, "#7f7f7f"),
    alpha=0.2
)

#title and Axis description
plt.title(f"Induction: {mood}", fontsize=18)
plt.ylabel(f"VAS {scale.replace('vas_scales_', '')}", fontsize=16)
plt.xlabel("")
plt.ylim(0, 100)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.xticks(rotation=45, ha='right')  # rotation=45 dreht die Labels um 45 Grad, ha='right' sorgt dafür, dass die Labels richtig ausgerichtet sind

#layout and saving
plt.tight_layout()

plt.savefig(f"Plots/{scale}_plot_Mood_{mood}_{model}.jpg", bbox_inches="tight")
plt.savefig(f"Plots_SVG/{scale}_plot_Mood_{mood}_{model}.svg", bbox_inches="tight")

plt.show()
