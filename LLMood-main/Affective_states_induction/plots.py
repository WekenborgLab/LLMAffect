#!/usr/bin/env python3
"""
Mood Induction Visualization Script

This script processes CSV data from mood induction experiments and creates
visualizations for VAS scales and state anxiety measures.

Usage:
    python plots.py --csv-path=/path/to/your/results.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process and visualize mood induction data.')
    parser.add_argument('--csv-path', required=True, help='Path to the CSV file')
    return parser.parse_args()


def setup_output_dirs(base_path):
    """Create output directories for plots."""
    plots_dir = os.path.join(base_path, "plots")
    svg_dir = os.path.join(base_path, "plots_svg")
    
    # Create directories if they don't exist
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    
    return plots_dir, svg_dir


def load_and_preprocess_data(csv_path):
    """Load and preprocess the data from CSV."""
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Mapping for reverse coding
    reverse_mapping = {1: 4, 2: 3, 3: 2, 4: 1}
    
    # Columns to reverse
    columns_to_reverse = [
        "state_anxiety_calm", "state_anxiety_secure", "state_anxiety_at_ease",
        "state_anxiety_rested", "state_anxiety_comfortable", "state_anxiety_confident",
        "state_anxiety_relaxed", "state_anxiety_content", "state_anxiety_happy", 
        "state_anxiety_cheerful"
    ]
    
    # Apply reverse coding
    for column in columns_to_reverse:
        if column in df.columns:
            df[column] = df[column].map(lambda x: reverse_mapping.get(x, x))
    
    # Calculate anxiety scores if all required columns exist
    anxiety_items = [col for col in df.columns if col.startswith("state_anxiety_")]
    if anxiety_items:
        df["state_anxiety_score_mean"] = df[anxiety_items].mean(axis=1)
        df["state_anxiety_score_sum"] = df[anxiety_items].sum(axis=1)
    
    return df


def plot_vas_scales(df, plots_dir, svg_dir):
    """Generate plots for VAS scales."""
    # Define mood parameters
    mood_colors = {
        "neutral": "k",
        "fear": "#713E80",
        "anxiety": "#E07233",
        "anger": "#C54028",
        "disgust": "#429235",
        "sadness": "#4293C2",
        "worry": "m",
        "stress": "darkred"
    }
    
    mood_order = ["neutral", "fear", "anxiety", "anger", "disgust", "sadness", "worry"]
    section_to_mood = dict(zip(range(7), mood_order))
    
    # Define prompt stages
    prompt_labels = {
        "0-0": "Baseline",
        "1-0": "After induction",
        "3-0": "After regulation"
    }
    relevant_prompts = ["0-0", "1-0", "3-0"]
    
    # Identify VAS scales
    vas_scales = [col for col in df.columns if "vas_scales" in col]
    if not vas_scales:
        print("No VAS scales found in the data")
        return
    
    # Group data and calculate statistics
    df_grouped = df.groupby(["promptid-trialid", "section_number"])[vas_scales].agg(["mean", "std", "count"]).reset_index()
    
    # Calculate confidence intervals
    for scale in vas_scales:
        df_grouped[f"{scale}_sem"] = df_grouped[(scale, "std")] / np.sqrt(df_grouped[(scale, "count")])
        df_grouped[f"{scale}_lower_ci"] = df_grouped[(scale, "mean")] - 1.96 * df_grouped[f"{scale}_sem"]
        df_grouped[f"{scale}_upper_ci"] = df_grouped[(scale, "mean")] + 1.96 * df_grouped[f"{scale}_sem"]
    
    # Rename columns for easier access
    df_grouped.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in df_grouped.columns]
    
    # Add labels for stages and moods
    df_grouped["prompt_stage"] = df_grouped["promptid-trialid"].map(prompt_labels)
    df_grouped["mood_label"] = df_grouped["section_number"].map(section_to_mood)
    
    # Filter to relevant prompt stages
    df_filtered = df_grouped[df_grouped["promptid-trialid"].isin(relevant_prompts)]
    
    # Generate plots for each VAS scale and mood
    for scale in vas_scales:
        for mood in mood_order:
            # Create a new figure
            plt.figure(figsize=(6, 4))
            
            # Filter by mood
            mood_df = df_filtered[df_filtered["mood_label"] == mood]
            if mood_df.empty:
                plt.close()
                continue
            
            # Create point plot
            sns.pointplot(
                data=mood_df,
                x="prompt_stage",
                y=f"{scale}_mean",
                color=mood_colors.get(mood, "gray"),
                errorbar=None,
                markers="o",
                linestyles="-"
            )
            
            # Add confidence interval bands
            plt.fill_between(
                mood_df["prompt_stage"],
                mood_df[f"{scale}_lower_ci"],
                mood_df[f"{scale}_upper_ci"],
                color=mood_colors.get(mood, "gray"),
                alpha=0.2
            )
            
            # Set titles and labels
            plt.title(f"{scale} – Moodinduction: {mood.capitalize()}", fontsize=18)
            plt.ylabel(f"{scale.replace('vas_scales_', '')} Score", fontsize=16)
            plt.ylim(0, 100)
            plt.tick_params(axis='both', which='major', labelsize=14)
            
            # Save plots
            plt.tight_layout()
            jpg_path = os.path.join(plots_dir, f"{scale}_plot_Mood_{mood}.jpg")
            svg_path = os.path.join(svg_dir, f"{scale}_plot_Mood_{mood}.svg")
            plt.savefig(jpg_path, bbox_inches="tight")
            plt.savefig(svg_path, bbox_inches="tight")
            plt.close()


def plot_anxiety_scores(df, plots_dir, svg_dir):
    """Generate plots for anxiety scores."""
    if "state_anxiety_score_sum" not in df.columns:
        print("State anxiety scores not found in the data")
        return
    
    # Define mood parameters
    mood_colors = {
        "neutral": "k",
        "fear": "#713E80",
        "anxiety": "#E07233",
        "anger": "#C54028",
        "disgust": "#429235",
        "sadness": "#4293C2",
        "worry": "m"
    }
    
    mood_order = ["neutral", "fear", "anxiety", "anger", "disgust", "sadness", "worry"]
    section_to_mood = dict(zip(range(7), mood_order))
    
    # Define prompt stages
    prompt_labels = {
        "0-0": "Baseline",
        "1-0": "After induction",
        "3-0": "After regulation"
    }
    relevant_prompts = ["0-0", "1-0", "3-0"]
    
    # Group data and calculate statistics
    df_anx_grouped = df.groupby(["promptid-trialid", "section_number"])["state_anxiety_score_sum"].agg(["mean", "std", "count"]).reset_index()
    
    # Calculate confidence intervals
    df_anx_grouped["sem"] = df_anx_grouped["std"] / np.sqrt(df_anx_grouped["count"])
    df_anx_grouped["lower_ci"] = df_anx_grouped["mean"] - 1.96 * df_anx_grouped["sem"]
    df_anx_grouped["upper_ci"] = df_anx_grouped["mean"] + 1.96 * df_anx_grouped["sem"]
    
    # Add labels for stages and moods
    df_anx_grouped["prompt_stage"] = df_anx_grouped["promptid-trialid"].map(prompt_labels)
    df_anx_grouped["mood_label"] = df_anx_grouped["section_number"].map(section_to_mood)
    
    # Filter to relevant prompt stages
    df_anx_plot = df_anx_grouped[df_anx_grouped["promptid-trialid"].isin(relevant_prompts)]
    
    # Generate plots for each mood
    for mood in mood_order:
        plt.figure(figsize=(6, 4))
        
        # Filter by mood
        mood_df = df_anx_plot[df_anx_plot["mood_label"] == mood]
        if mood_df.empty:
            plt.close()
            continue
        
        # Create point plot
        sns.pointplot(
            data=mood_df,
            x="prompt_stage",
            y="mean",
            color=mood_colors.get(mood, "gray"),
            errorbar=None,
            markers="o",
            linestyles="-"
        )
        
        # Add confidence interval bands
        plt.fill_between(
            mood_df["prompt_stage"],
            mood_df["lower_ci"],
            mood_df["upper_ci"],
            color=mood_colors.get(mood, "gray"),
            alpha=0.2
        )
        
        # Set titles and labels
        plt.title(f"State Anxiety – Moodinduction: {mood.capitalize()}", fontsize=18)
        plt.ylabel("State Anxiety Score (Sum)", fontsize=16)
        plt.ylim(0, df_anx_plot["mean"].max() + 5)
        plt.tick_params(axis='both', which='major', labelsize=14)
        
        # Save plots
        plt.tight_layout()
        jpg_path = os.path.join(plots_dir, f"state_anxiety_plot_Mood_{mood}.jpg")
        svg_path = os.path.join(svg_dir, f"state_anxiety_plot_Mood_{mood}.svg")
        plt.savefig(jpg_path, bbox_inches="tight")
        plt.savefig(svg_path, bbox_inches="tight")
        plt.close()


def generate_plots(csv_path):
    """
    Main function to generate plots that can be imported and called from other scripts.
    
    Args:
        csv_path (str): Path to the CSV file with the data
        
    Returns:
        tuple: Paths to the plots directory and SVG directory
    """
    # Get base directory from CSV path
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    
    # Setup output directories
    plots_dir, svg_dir = setup_output_dirs(base_dir)
    print(f"Output directories created: {plots_dir} and {svg_dir}")
    
    # Load and preprocess data
    df = load_and_preprocess_data(csv_path)
    
    # Generate plots
    plot_vas_scales(df, plots_dir, svg_dir)
    plot_anxiety_scores(df, plots_dir, svg_dir)
    
    print(f"Plots generated successfully in {plots_dir} and {svg_dir}")
    
    return plots_dir, svg_dir


def main():
    """Command-line entry point."""
    # Parse command line arguments
    args = parse_arguments()
    generate_plots(args.csv_path)


if __name__ == "__main__":
    main()
