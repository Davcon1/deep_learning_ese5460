# combine_datasets.py
import torch
import os
import argparse
from pathlib import Path


def combine_trajectory_files(input_dir, output_file, pattern="traj_*.pt", max_iteration=None, exclude_combined=True):
    '''
    Used for combining multiple trajectories that have been built together
    '''
    input_path = Path(input_dir)
    dataset_files = sorted(input_path.glob(pattern))
    
    # filter out combined dataset files if requested
    if exclude_combined:
        filtered_files = []
        for f in dataset_files:
            if not f.name.startswith("combined_dataset"):
                filtered_files.append(f)
            else:
                print(f"{f.name} (combined dataset)")
        dataset_files = filtered_files
    
    # Filter by the max iteration length
    if max_iteration is not None:
        filtered_files = []
        for f in dataset_files:
            # Extract iteration number from filename (e.g., traj_dance1_subject2_iter2.pt)
            name = f.stem
            if "_iter" in name:
                try:
                    iter_num = int(name.split("_iter")[-1])
                    if iter_num <= max_iteration:
                        filtered_files.append(f)
                except ValueError:
                    # If can't parse iteration number, include it
                    filtered_files.append(f)
            else:
                # Files without iteration number are included (backward compatibility)
                filtered_files.append(f)
        dataset_files = filtered_files
    
    if not dataset_files:
        raise ValueError(f"No trajectory files found in {input_dir} with pattern {pattern}")
    
    print(f"Found {len(dataset_files)} trajectory files:")
    for f in dataset_files:
        print(f"  - {f.name}")
    
    # Load all datasets
    all_data = {}
    total_samples = 0
    
    for i, filepath in enumerate(dataset_files):
        print(f"\nLoading {filepath.name}...")
        try:
            data = torch.load(filepath, map_location='cpu')
        except Exception as e:
            print(f"Failed to load {filepath.name}: {e}")
            continue
        
        if i == 0:
            for key in data.keys():
                if key != 'metadata':
                    all_data[key] = []
        
        for key, value in data.items():
            if key == 'metadata':
                continue
                
            if isinstance(value, torch.Tensor):
                # Handle different tensor shapes
                if len(value.shape) == 3:
                    flattened = value.reshape(-1, value.shape[-1])
                else:
                    flattened = value
                
                all_data[key].append(flattened)
                
                if key == 'obs':
                    total_samples += flattened.shape[0]
                    print(f"  Added {flattened.shape[0]} samples")
            else:
                print(f" Skipping non-tensor key: {key}")
    
    if not all_data:
        raise ValueError("No valid data loaded from trajectory files")
    
    # Concatenate all data
    combined = {}
    for key, value_list in all_data.items():
        if value_list:  # Check if list is not empty
            combined[key] = torch.cat(value_list, dim=0)
            print(f"\n{key}: {combined[key].shape}")
        else:
            print(f"\n No data for key: {key}")
    
    # Add metadata
    combined['metadata'] = {
        'num_files': len(dataset_files),
        'total_samples': total_samples,
        'source_files': [f.name for f in dataset_files],
        'max_iteration': max_iteration,
        'excluded_combined': exclude_combined
    }
    
    # Save combined dataset
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    torch.save(combined, output_file)
    print(f"\n Combined dataset saved to {output_file}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Files used: {len(dataset_files)}")
    
    return combined


def combine_specific_iterations(input_dir, output_file, iterations=None):
    '''
    Can combine specific iterations with equal weightings
    '''
    if iterations is None:
        iterations = []
    
    input_path = Path(input_dir)
    all_files = []
    
    for iteration in iterations:
        # Pattern for specific iteration
        pattern = f"traj_*_iter{iteration}.pt"
        files = sorted(input_path.glob(pattern))
        
        # Filter out combined dataset files
        files = [f for f in files if not f.name.startswith("combined_dataset")]
        
        print(f"Iteration {iteration}: Found {len(files)} files")
        all_files.extend(files)
    
    if not all_files:
        raise ValueError(f"No trajectory files found for iterations {iterations}")
    
    print(f"\nTotal files to combine: {len(all_files)}")
    
    # Now use the main combining function
    return combine_trajectory_files(
        input_dir=input_dir,
        output_file=output_file,
        pattern="*",
        max_iteration=None,
        exclude_combined=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine trajectory datasets")
    parser.add_argument("--input_dir", type=str, default="../datasets", help="Directory containing trajectory files")
    parser.add_argument("--output", type=str, default="../datasets/combined_dataset.pt", help="Output file path")
    parser.add_argument("--pattern", type=str, default="traj_*.pt", help="Glob pattern for finding files")
    parser.add_argument("--max_iteration", type=int, default=None, help="Only include files up to this iteration")
    parser.add_argument("--exclude_combined", action="store_true", default=True, help="Exclude combined_dataset files")
    parser.add_argument("--iterations", type=str, default=None, help="Comma-separated list of iterations to include (e.g., 0,1,2,3)")
    
    args = parser.parse_args()
    
    # Parse iterations if provided
    iterations_list = None
    if args.iterations:
        iterations_list = [int(x.strip()) for x in args.iterations.split(",")]
        print(f"Combining specific iterations: {iterations_list}")
        
        combine_specific_iterations(
            input_dir=args.input_dir,
            output_file=args.output,
            iterations=iterations_list
        )
    else:
        combine_trajectory_files(
            input_dir=args.input_dir,
            output_file=args.output,
            pattern=args.pattern,
            max_iteration=args.max_iteration,
            exclude_combined=args.exclude_combined
        )