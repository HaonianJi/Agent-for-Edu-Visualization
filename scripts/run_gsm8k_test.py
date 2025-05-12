import os
import json
import argparse
from datetime import datetime
import sys

# Add project root to Python path to allow direct script execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.teach_agents_intermediate_system import TeachAgentsIntermediateSystem
from config.system_config_loader import SystemConfigLoader

def main():
    parser = argparse.ArgumentParser(description="Run GSM8K tests using TeachAgentsIntermediateSystem.")
    parser.add_argument(
        "--data_file_path",
        type=str,
        default="mydatasets/train.jsonl",
        help="Path to the GSM8K JSONL data file (relative to project root)."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/multi_agents/education.yaml",
        help="Path to the multi-agent system config YAML file (relative to project root)."
    )
    parser.add_argument(
        "--output_dir_base",
        type=str,
        default="outputs/gsm8k_tests",
        help="Base directory to save test outputs (relative to project root)."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process from the data file (processes all if not specified)."
    )
    parser.add_argument(
        "--question_field",
        type=str,
        default="question",
        help="The field name in the JSONL file that contains the question text."
    )

    args = parser.parse_args()

    # Construct absolute paths from project root
    abs_data_file_path = os.path.join(project_root, args.data_file_path)
    abs_config_path = os.path.join(project_root, args.config_path)
    abs_output_dir_base = os.path.join(project_root, args.output_dir_base)

    if not os.path.exists(abs_data_file_path):
        print(f"Error: Data file not found at {abs_data_file_path}")
        return

    if not os.path.exists(abs_config_path):
        print(f"Error: Config file not found at {abs_config_path}")
        return

    try:
        # Load system configuration
        cfg_loader = SystemConfigLoader(config_path_input=abs_config_path)
        system_config = cfg_loader.get_config()
        
        # Initialize the TeachAgentsIntermediateSystem
        # The system will create its own output subdirectories based on base_run_output_dir
        teach_system = TeachAgentsIntermediateSystem(system_config=system_config)
        
        print(f"TeachAgentsIntermediateSystem initialized with config: {abs_config_path}")
        print(f"Processing data from: {abs_data_file_path}")
        print(f"Saving outputs to base directory: {abs_output_dir_base}")

        with open(abs_data_file_path, 'r') as f:
            lines = f.readlines()

        samples_to_process = lines
        if args.num_samples is not None and args.num_samples > 0:
            samples_to_process = lines[:args.num_samples]
            print(f"Processing the first {args.num_samples} samples.")

        for i, line in enumerate(samples_to_process):
            try:
                data_item = json.loads(line)
                question = data_item.get(args.question_field)

                if not question:
                    print(f"Warning: Skipping line {i+1} due to missing '{args.question_field}' field.")
                    continue

                print(f"\nProcessing sample {i+1}/{len(samples_to_process)}: \"{question[:100]}...\"")

                # Create a unique output directory for this specific question's run
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Sanitize question for directory name (simple version)
                query_slug = "".join(c if c.isalnum() else "_" for c in question[:50]).strip("_")
                run_output_dir_name = f"{timestamp}_{i+1}_{query_slug}"
                question_run_output_dir = os.path.join(abs_output_dir_base, run_output_dir_name)
                
                # TeachAgentsIntermediateSystem expects base_run_output_dir for its internal saving structure
                teach_system.base_run_output_dir = question_run_output_dir # Set it for each run

                # Predict
                # The system now handles its own saving based on the base_run_output_dir
                results = teach_system.predict(question=question) 

                # The 'results' might contain the final plan, depending on predict's return value
                # For now, we rely on the system's internal saving.
                # We can print a part of the result if needed.
                final_plan_summary = "Final plan generated."
                if results and isinstance(results, dict) and "final_combined_plan" in results:
                    final_plan_summary = results["final_combined_plan"][:200] + "..."
                
                print(f"Finished processing sample {i+1}. Outputs saved in: {question_run_output_dir}")
                print(f"Final plan summary: {final_plan_summary}")

            except json.JSONDecodeError:
                print(f"Warning: Skipping line {i+1} due to JSON decoding error.")
            except Exception as e:
                print(f"Error processing sample {i+1} ('{question[:50]}...'): {e}")
                # Optionally, decide if you want to continue or stop on error
                # continue 

        print("\nFinished processing all specified samples.")

    except Exception as e:
        print(f"An critical error occurred: {e}")

if __name__ == "__main__":
    main()
