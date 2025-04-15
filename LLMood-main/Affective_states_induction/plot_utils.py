# json_analysis.py
import os
import json
import pandas as pd
import plotly.express as px


def flatten_json(nested_json, separator='_'):
    """Flatten a nested JSON object."""
    out = {}

    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + separator)
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, name + str(i) + separator)
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out


def process_json_data(results, filename="results.csv"):
    """Process JSON data stored as a list of tuples and save to CSV."""
    all_data = []
    all_keys = set()

    for run_id, (prompts_dict, iteration_number, section_number) in enumerate(results):  
        if not isinstance(prompts_dict, dict):
            print(f"Skipping invalid entry at index {run_id}: {prompts_dict}")
            continue

        for prompt_index, (prompt, result_json_string_list) in enumerate(prompts_dict.items()):
            if not isinstance(result_json_string_list, list):
                result_json_string_list = [result_json_string_list]

            for trial_id, result_json_string in enumerate(result_json_string_list):
                try:
                    result_json = json.loads(result_json_string) if isinstance(result_json_string, str) else result_json_string
                    flattened = flatten_json(result_json)
                    flattened["run_id"] = run_id
                    flattened["iteration_id"] = iteration_number
                    flattened["section_number"] = section_number
                    flattened["promptid-trialid"] = f"{prompt_index}-{trial_id}"
                    flattened["prompt"] = prompt
                    all_keys.update(flattened.keys())
                    all_data.append(flattened)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in prompt '{prompt}':", result_json_string)

    df = pd.DataFrame(all_data)
    df = df[[col for col in all_keys]]  # Optionale Spaltenreihenfolge
    df.to_csv(filename, index=False)
    print(f"Data successfully saved to {filename}")
    return df


def plot_vas_scales(df, prompt_labels, vas_scales, save_dir=None):
    """Create facet line plots for VAS scales and optionally save them."""
    df_plot = df.copy()
    df_plot["promptid-trialid"] = df_plot["promptid-trialid"].map(prompt_labels)

    df_melted = df_plot.melt(
        id_vars=["promptid-trialid", "run_id"],
        value_vars=vas_scales,
        var_name="VAS_Skala",
        value_name="Wert"
    )

    fig = px.line(
        df_melted,
        x="promptid-trialid",
        y="Wert",
        color="run_id",
        facet_col="VAS_Skala",
        facet_col_wrap=2,
        markers=True,
        title="Verlauf der VAS-Skalen über die Messzeitpunkte"
    )

    fig.update_layout(
        xaxis_title="Messzeitpunkte",
        yaxis_title="VAS-Wert (0-100)",
        xaxis=dict(tickmode="array", tickvals=list(prompt_labels.values())),
        height=800,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        html_path = os.path.join(save_dir, "vas_scales_plot.html")
        png_path = os.path.join(save_dir, "vas_scales_plot.png")

        fig.write_html(html_path)
        fig.write_image(png_path, scale=2)
        print(f"Plot saved to {html_path} and {png_path}")

    fig.show()
