from argparse import ArgumentParser
from datetime import datetime
from typing import Type
import openai
import json
import os
import yaml
from pydantic import BaseModel
from tqdm import tqdm
from pydantic_schemas import EmotionalStateAssessment
from utils import read_moods_and_prompts, save_to_docx, query_LLM
from plot_utils import process_json_data
from plots import generate_plots


def run_experiment_1(
    client: openai.OpenAI, 
    api_type: str, 
    temperature: int, 
    model: str, 
    experiment_name: str, 
    moods_prompts: list[list[str]], 
    pydantic_schema: Type[BaseModel] = None, 
    iterations=1, 
    results_path: str = None, 
    parallel: int = 1,
    verbose: bool = False
) -> None:
    """
    Runs the experiment for the given moods and prompts.
    
    Args:
        client: The OpenAI client.
        api_type: The type of API to use. Mainly to correctly run structured outputs.
        temperature: The temperature setting for the model.
        model: The model to use.
        experiment_name: The name of the experiment.
        moods_prompts: A list of lists, where each inner list contains prompts for a mood.
        pydantic_schema: Optional Pydantic schema for structured output.
        iterations: The number of iterations to run the experiment.
        output_file: The file to save the results to.
        parallel: The number of parallel requests to make. (Default = 1, which means no parallel requests)
    """
    import concurrent.futures
    from functools import partial
    
    def run_single_iteration(iteration, moods_prompts, client, api_type, temperature, model, pydantic_schema, verbose):
        """Run a single iteration of the experiment"""
        #TODO parallelize the moods too

        results = []
        for mood_number, prompts in enumerate(moods_prompts):
            mood_results = {}
            messages = []
            for prompt_number, prompt in enumerate(prompts):
                print(f"Running iteration {iteration} for mood {mood_number} and prompt {prompt_number}...")
                messages.append({"role": "user", "content": prompt})
                if prompt not in mood_results:
                    mood_results[prompt] = []
                if prompt_number != 2:
                    response = query_LLM(
                        client=client, 
                        api_type=api_type, 
                        temperature=temperature, 
                        messages=messages, 
                        model=model, 
                        pydantic_schema=pydantic_schema,
                        verbose=verbose
                    )
                    messages.append({"role": "assistant", "content": json.dumps(response)})
                    mood_results[prompt].append(response)
            results.append((mood_results, iteration, mood_number))
        return results
    
    all_results = []
    
    # Create a partial function with all parameters except iteration
    run_iteration = partial(
        run_single_iteration, 
        moods_prompts=moods_prompts, 
        client=client, 
        api_type=api_type, 
        temperature=temperature, 
        model=model, 
        pydantic_schema=pydantic_schema,
        verbose=verbose
    )
    
    # Execute iterations sequentially or in parallel based on the 'parallel' parameter
    if parallel <= 1:
        if verbose:
            print("Running iterations sequentially...")
        # Run iterations sequentially with progress bar
        for iteration in tqdm(range(iterations)):
            print(f"Running iteration {iteration + 1}/{iterations}")
            iteration_results = run_iteration(iteration)
            all_results.extend(iteration_results)
    else:
        if verbose:
            print(f"Running iteractions parallel with {parallel} workers...")
        # Run iterations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(parallel, iterations)) as executor:
            # Submit all iteration tasks
            future_to_iteration = {
                executor.submit(run_iteration, iteration): iteration 
                for iteration in range(iterations)
            }
            
            # Process results as they complete with a progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_iteration), total=iterations):
                iteration_results = future.result()
                all_results.extend(iteration_results)
    
    # Save results to file if specified

    csv_name = None
    if results_path:
        output_filename = os.path.join(results_path, "results.docx")
        save_to_docx(all_results, output_filename, model_name=model, save_as_markdown=True)
        
        csv_name = os.path.join(results_path, "results.csv")
    
    df = process_json_data(all_results, filename=csv_name)

    # prompt_labels = {
    #     "0-0": "Neutral",
    #     "1-0": "Induction",
    #     "2-0": "Intervention"
    # }

    # TODO automatic generation
    # vas_scales = ["vas_scales_fear", "vas_scales_happiness", "vas_scales_disgust", "vas_scales_anger", "vas_scales_worry"]

    # plot_vas_scales(df, prompt_labels, vas_scales, save_dir=results_path)

    generate_plots(csv_name)

    return all_results

def main(args: ArgumentParser) -> None:
    """
    Main entry point for the script.

    Args:
        args: Parsed command line arguments.

    Raises:
        ValueError: If the API key is invalid.
        ValueError: If the API base is invalid.
        ValueError: If the selected model is not found in the list of models.
    """
    assert(args.api_type in ["openai", "vllm", "openai-compatible"])
    if args.api_type == "openai":
        client = openai.OpenAI(api_key=args.api_key)
    else:
        client = openai.OpenAI(api_key=args.api_key, base_url=args.api_base)

    try:
        models = client.models.list()
    except openai.AuthenticationError as e:
        raise ValueError("Invalid API key!", e)
    except Exception as e:
        raise ValueError("Unknown error!", e)
                        
    models = [model.id for model in models]
    if args.model not in models:
        raise ValueError(f"Selected model {args.model} not found in list of models: {', '.join(models)}")
    
    #read in prompts
    #TODO Put mood names in dict
    moods_prompts = read_moods_and_prompts(args.moods)
    
    # create results directory consisting of experiment name and timestamp
    results_path = os.path.join(args.results_dir, f"{args.experiment_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    os.makedirs(results_path, exist_ok=True)

    # Save Experiment Metadata
    metadata = {
        "experiment_name": args.experiment_name,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "api_type": args.api_type,
        "api_base": args.api_base if args.api_base else "default",
        "temperature": args.temperature,
        "iterations": args.iterations,
        "moods": args.moods,
        "parallel": args.parallel,
        "verbose": args.verbose,
    }

    with open(os.path.join(results_path, "metadata.yaml"), "w") as f:
        yaml.dump(metadata, f)

    if args.experiment_type == "experiment1":
        #run experiment 1
        run_experiment_1(client=client, api_type=args.api_type, temperature=args.temperature, model=args.model, experiment_name=args.experiment_name, moods_prompts=moods_prompts, iterations=args.iterations, verbose=args.verbose, pydantic_schema=EmotionalStateAssessment, results_path=results_path, parallel=args.parallel)
    else:
        raise ValueError(f"Unknown experiment type {args.experiment_type}")


args = ArgumentParser()

model_settings = args.add_argument_group("Model Settings")
model_settings.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
model_settings.add_argument("--api_type", type=str, default="openai", choices=["openai", "vllm", "openai-compatible"])
model_settings.add_argument("--api_key", type=str, default="DUMMY_API_KEY")
model_settings.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
model_settings.add_argument("--temperature", type=float, default=0.5)
model_settings.add_argument("--parallel", type=int, default=1)

experiment_settings = args.add_argument_group("Experiment Settings")
experiment_settings.add_argument("--experiment-type", type=str, choices=["experiment1", "experiment2"], default="experiment1")
experiment_settings.add_argument("--experiment_name", type=str, default="Test")
experiment_settings.add_argument("--iterations", type=int, default=1)
experiment_settings.add_argument("--moods", type=str, nargs="+", default=["Neutral", "Fear", "Anxiety", "Anger", "Disgust", "Sadness", "Worry"])
experiment_settings.add_argument("--results-dir", type=str, default="results")

general_settings = args.add_argument_group("General Settings")
general_settings.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")

args = args.parse_args()

main(args)
