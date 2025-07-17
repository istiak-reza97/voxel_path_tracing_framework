import h5py
import numpy as np
import pandas as pd
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

# Critical HDF5 tuning
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def process_h5_file(filepath):
    """Single-pass file processor with optimized HDF5 access"""
    start_time = time.time()
    file_size = os.path.getsize(filepath) / (1024**3)  # GB
    
    with h5py.File(filepath, 'r', libver='latest', swmr=True) as f:
        # Pre-allocate arrays (we'll resize later)
        max_possible = len(f)
        times = np.empty(max_possible, dtype=np.float32)
        count = 0
        
        for trial_name in f:
            if not trial_name.startswith('trial_'):
                continue
                
            try:
                trial = f[trial_name]
                if trial.attrs.get('status') == 'success':
                    times[count] = trial.attrs['total_transit_time']+trial.attrs['time_art_jump']+trial.attrs['time_ven_jump']
                    count += 1
            except (KeyError, AttributeError):
                continue
    
    proc_time = time.time() - start_time
    
    if count == 0:
        print(f"No successful trials in {os.path.basename(filepath)}")
        return 0, np.nan, np.nan
    
    # Resize arrays to actual count
    times = times[:count]
    
    print(f"Processed {os.path.basename(filepath)} ({file_size:.1f}GB, {count} trials) in {proc_time:.1f}s")
    return count, np.mean(times), np.median(times)

def analyze_transit_times(file_pattern="FINAL2_transit_results*.h5"):
    all_files = sorted(glob.glob(file_pattern))
    if not all_files:
        raise ValueError("No files found matching pattern")

    # Cluster-optimized settings
    max_workers = min(10, os.cpu_count())
    chunk_size = max(1, len(all_files) // (max_workers * 2))  # Now properly used
    
    results = {
        'counts': np.zeros(len(all_files), dtype=np.int32),
        'mean_times': np.zeros(len(all_files), dtype=np.float32),
        'median_times': np.zeros(len(all_files), dtype=np.float32),
        'file_names': []
    }
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Using chunksize properly with map()
        future = executor.map(process_h5_file, all_files, chunksize=chunk_size)
        
        for i, (count, mean, median) in enumerate(future):
            results['file_names'].append(os.path.basename(all_files[i]))
            if count > 0:
                results['counts'][i] = count
                results['mean_times'][i] = mean
                results['median_times'][i] = median
    
    # Filter out empty results
    valid_mask = results['counts'] > 0
    for key in ['counts', 'mean_times', 'median_times']:
        results[key] = results[key][valid_mask]
    results['file_names'] = [n for i, n in enumerate(results['file_names']) if valid_mask[i]]
    
    print(f"\nProcessed {len(all_files)} files in {time.time()-start_time:.1f}s")
    return results

if __name__ == "__main__":
    start_time = time.time()
    results = analyze_transit_times()
    
    # Save comprehensive results
    np.savez('transit_time_results.npz',
             counts=results['counts'],
             mean_times=results['mean_times'],
             median_times=results['median_times'],
             file_names=results['file_names'])
    
    print(f"Total runtime: {(time.time()-start_time)/60:.2f} mins")
    print("\nResults Summary:")
    for name, count, mean, median in zip(results['file_names'],
                                        results['counts'],
                                        results['mean_times'],
                                        results['median_times']):
        print(f"{name}: {count} trials | Mean: {mean:.4f} | Median: {median:.4f}")
        
'''
data = np.load('transit_time_results.npz', allow_pickle=True)
# Create a dictionary from the .npz data
data_dict = {key: data[key] for key in data.files}

# Convert to DataFrame
df = pd.DataFrame(data_dict)
'''