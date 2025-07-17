import numpy as np
import pandas as pd
from multiprocessing import Pool, Process
from collections import defaultdict 
from numba import njit
import time
import h5py
import copy
import pickle
import argparse

# Define the parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--filename', type=str, default="test", help='Path to the file')
# parser.add_argument('--csv_x', type=float, default=0.5, help='Threshold value')

# Parse the arguments
args = parser.parse_args()

# Save arguments as variables
file_path = args.filename
 

# Constants
dz = 0.333e-3  # m
dx = dy = 6.4e-5  # m
dV = dx * dy * dz
dz_sq = dz**2
dx_sq = dx**2

e = 0.01  # m
myu = 3e-3  # Pa·s
Ka = 1e-12  # m²
Kv = 5e-10  # m²
a = 1e-6  # perfusion coefficient
gamma_v = 1e-14  # m²
delta = 0.5 * 7.5e-6 #m avg dia of wbc(6-9 *1e-6 m)


un_t = 555723  # total nodes
un_a = 121  # arterial nodes
inlets = [0, 24]
outlets = [0, 20]

stencil = np.array([
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
], dtype=np.int64)

LA_stencil = np.array([
    (dx, dy, dz), (dx, dy, dz),  # x-dir
    (dy, dz, dx), (dy, dz, dx),  # y-dir
    (dz, dx, dy), (dz, dx, dy)   # z-dir
], dtype=np.float64)

# Numba-optimized functions

@njit(fastmath=True, cache=True)
def SoI(center, point):
    # Direct unpacking (faster than tuple indexing)
    # x0, y0, z0 = center
    # x1, y1, z1 = point
    # center_arr = np.asarray(center)
    # point_arr = np.asarray(point)
    
    # Compute squared differences
    # dx = x0 - x1
    # dy = y0 - y1
    # dz = z0 - z1
    diff = point - center
    
    #print(point, center, diff)
    
    # Weight by voxel dimensions and sum
    return np.sqrt(diff[0]*diff[0] * dx_sq + 
                  diff[1]*diff[1]*dx_sq +  # Using dx_sq since dy = dx
                  diff[2]*diff[2] * dz_sq)

@njit(fastmath=True, cache=True)
def eta(x, e, C):
    x_over_e = x / e
    if x_over_e < 1.0:
        x_over_e_sq = x_over_e * x_over_e  # Faster than x_over_e**2
        inv_denominator = 1.0 / (x_over_e_sq - 1.0)
        return C[0] * np.exp(inv_denominator)
    return 0.0


@njit(fastmath=True, cache=True)
def kappa(d, l):
    return np.pi * (d * dx)**4 / (128 * myu * l * dx)

@njit(fastmath=True, cache=True)
def dLength(pos1, pos2):
    """Numba-optimized distance calculator"""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    dz = pos2[2] - pos1[2]
    
    for i in range(len(stencil)):
        if (dx == stencil[i,0] and 
            dy == stencil[i,1] and 
            dz == stencil[i,2]):
            return LA_stencil[i, 0]  # Return length component
    
    return 0.0

@njit(fastmath=True, cache=True)
def Time_art_vein(diameter, p1, p2, l):
    sampled_r = RandSamp4Vel(diameter/2)
    #print(f'sampled radius:', sampled_r)
    return (dx * l)**2 / ((p1-p2) * ((dx*(diameter/2))**2 - (dx*sampled_r)**2)/(4*myu))
    #return (dx * l)**2 / ((p1-p2) * ((dx*(diameter/2))**2 - (dx*RandSamp4Vel(diameter/2))**2)/(4*myu))

@njit(fastmath=True, cache=True)
def Time_tissue(ds, p1, p2, K):
    return ds**2 / (K*(p1-p2)/myu)

@njit(fastmath=True, cache=True)
def RandSamp4Vel(R):
    #return R * np.sqrt(np.random.uniform(0.0, 1))
    return (R-delta) * np.sqrt(np.random.uniform(0.0, 1))    #adjusted for wbc radius

# Helper functions
def initialize_pressure_domains(dom, c_dom, X):
    
    nx, ny, nz = dom.shape
    press_dom_a = np.full((nx, ny, nz), np.nan, dtype=np.float32)
    press_dom_v = np.full((nx, ny, nz), np.nan, dtype=np.float32)
    
    # Create masks for tissue voxels (z=1-3 as per your original code)
    tissue_mask = (dom == 1) & (np.arange(nz)[None, None, :] >= 1) & (np.arange(nz)[None, None, :] <= 3)
    
    # Vectorized assignment
    press_dom_a[tissue_mask] = X[c_dom[tissue_mask]]
    press_dom_v[tissue_mask] = X[c_dom[tissue_mask] + 555723]  # Your venous offset
    
    print("initialize_pressure_domains checks")
    
    return press_dom_a, press_dom_v

def get_vessel_element(u, v, element_df):
    mask = ((element_df['initial node'] == u) & (element_df['final node'] == v)) | \
           ((element_df['final node'] == u) & (element_df['initial node'] == v))
           
    #print("get_vessel_element checks")
    return element_df.loc[mask].iloc[0]

def make_adjacency_dict(element_df):
    adj_dict = {}
    for _, row in element_df.iterrows():
        u, v = int(row['initial node']), int(row['final node'])
        adj_dict.setdefault(u, []).append(v)
        adj_dict.setdefault(v, []).append(u)
    print("make_adjacency_dict checks") 
    return adj_dict

def kappa_values_calculator(elememt_df):
    kappa_values = {}
    for u, v, d, l in zip(
            elememt_df['initial node'].values,
            elememt_df['final node'].values,
            elememt_df['diameter'].values,
            elememt_df['length'].values):
        k = kappa(d, l)
        kappa_values[(u, v)] = kappa_values[(v, u)] = k
    
    print("kappa_values_calculator checks")    
    
    return kappa_values

def precompute_flows_aTerm_to_Vox(a_out_nodes, nbr_a, a_element, kappa_values, X, Ca, e):
    """Precompute flow rates from terminal arteries to tissue voxels"""
    aflow = {}
    art_term_indices = [
        a_element[a_element['final node'] == node].index[0] 
        for node in a_out_nodes
    ]
    
    for i, node in enumerate(a_out_nodes):
        # Get artery segment properties
        element = a_element.loc[art_term_indices[i]]
        k1 = kappa_values[(element['initial node'], element['final node'])]
        x0, y0, z0 = a_out_nodes[node]  # Assuming a_out_nodes is {node: (x,y,z)}
        
        # Calculate flow to each neighboring voxel
        flows = []
        #eta_i=[]
        for j, voxel in enumerate(nbr_a[i]):
            s = SoI(np.asarray([y0, x0, z0]), np.asarray(voxel))
            n_ex = eta(s, e, Ca[i])
            #print(np.asarray([y0, x0, z0]), np.asarray(voxel), s,  n_ex )
            #eta_i.append(n_ex)
            G = k1 * n_ex * (
                X[2*un_t + element['initial node']] - 
                X[2*un_t + element['final node']]
            )
            flows.append(G)
        #print(i, np.sum(eta_i))
        aflow[(node)] = flows
    
    print("precompute_flows_aTerm_to_Vox checks")
    
    return aflow

def precompute_flows_Vox_to_vTerm(v_out_nodes, nbr_v, v_element, kappa_values, X, Cv, e):
    vflow = {node: [] for node in v_out_nodes}    # Original vflow structure {node_id: [flow1, flow2,...]}
    reversemap_vox_flow = defaultdict(list)  # New mapping: {voxel: [(node_id, flow), ...]}
    
    
    for i, node in enumerate(v_out_nodes.keys()):
        # Get artery segment properties
        element = v_element.loc[v_element['final node'] == node].iloc[0]
        k2 = kappa_values[(element['initial node'], element['final node'])]
        x0, y0, z0 = v_out_nodes[node]  # Assuming a_out_nodes is {node: (x,y,z)}
        
        # Calculate flow to each neighboring voxel
        flows = []
        
        
        for j, voxel in enumerate(nbr_v[i]):
            s = SoI(np.asarray([y0, x0, z0]), np.asarray(voxel))
            n_ex = eta(s, e, Cv[i])
            G = k2 * n_ex * (
                X[2*un_t + un_a + element['final node']] - 
                X[2*un_t + un_a + element['initial node']]      
            )
            vflow[node].append(G)  
            reversemap_vox_flow[tuple(voxel)].append((node, G))
        
        
    
    print("precompute_flows_Vox_to_vTerm checks")
        
    return vflow, reversemap_vox_flow

def precompute_tissue_movement_vectorized(dom, press_dom, K):
    
    # Create padded versions for boundary handling
    dom_pad = np.pad(dom, ((1,1),(1,1),(1,1)), mode='constant')
    press_pad = np.pad(press_dom, ((1,1),(1,1),(1,1)), mode='constant', constant_values=np.nan)
    
    # Prepare stencil directions (6 neighbors)
    stencil = np.array([
        [1,0,0], [-1,0,0],
        [0,1,0], [0,-1,0],
        [0,0,1], [0,0,-1]
    ])
    
    # Get all tissue voxel coordinates
    tissue_coords = np.argwhere(dom == 1)
    n_voxels = len(tissue_coords)
    #print('number of voxels: ', n_voxels)
    
    # Initialize output structures
    valid_moves = np.empty((n_voxels, 6, 3), dtype=np.int16)
    flows = np.empty((n_voxels, 6), dtype=np.float32)
    valid_mask = np.zeros((n_voxels, 6), dtype=bool)
    
    # Vectorized neighbor calculation
    neighbors = tissue_coords[:, None, :] + stencil[None, :, :]
    #print('neighbor shape: ',neighbors.shape)
    #print(neighbors[:4,...])
    #assert neighbors.shape[0] == len(tissue_coords) == 555723
    
    # Check valid moves (in tissue domain)
    in_domain = (neighbors >= 0) & (neighbors < np.array(dom.shape))
    is_tissue = dom_pad[
        neighbors[...,0]+1, 
        neighbors[...,1]+1, 
        neighbors[...,2]+1
    ] == 1
    valid = in_domain.all(axis=-1) & is_tissue
    
    # Calculate pressure gradients
    p_center = press_dom[
        tissue_coords[:,0], 
        tissue_coords[:,1], 
        tissue_coords[:,2]
    ][:, None]
    
    #print("p_center shape:",  p_center.shape)
    
    p_neighbors = press_pad[
        neighbors[...,0]+1,
        neighbors[...,1]+1,
        neighbors[...,2]+1
    ]
    
    #print("p_neighbors shape:", p_neighbors.shape)
    
    dp = p_center - p_neighbors
    #print("dp shape:", dp.shape)
    downhill = (dp > 0) & valid
    
    masked_dp = np.where(downhill, dp, 0)
    
    #print('downhill passed.')
    
    # Calculate flows for valid moves
    LA = np.array([
        [dx, dy, dz], [dx, dy, dz],  # x-dir
        [dy, dz, dx], [dy, dz, dx],   # y-dir
        [dz, dx, dy], [dz, dx, dy]    # z-dir
    ], dtype=np.float32)
    
    cross_areas = np.array([LA[i,1]*LA[i,2] for i in range(6)])
    lengths = np.array([LA[i,0] for i in range(6)])
    
    
    flows = (K/myu) * cross_areas * masked_dp / lengths  
    
    #print('flows_downhill passed.')
    
    # Store results
    valid_moves[downhill] = neighbors[downhill]
    valid_mask = downhill
    
    # Convert to dictionary format
    tissue_move_data = {
        tuple(coord): (
            valid_moves[i, valid_mask[i]].tolist(),
            flows[i, valid_mask[i]].tolist()
        )
        for i, coord in enumerate(tissue_coords)
    }
    
    print("precompute_tissue_movement_vectorized checks")
    
    return tissue_move_data

def select_tissue_entry(term_node, nbr_dict, a_out_nodes, aflow, X, press_dom):
    """Selects a tissue entry point from a terminal arterial node"""
    # Find which terminal node we're working with
    
    candidate_voxels = nbr_dict[term_node]
   
    
    flows = np.array([aflow[term_node][i] for i in range(len(candidate_voxels))])
    #print(flows[:10])
    probs = flows / flows.sum()
    
    satisfy = False
    while satisfy == False:
    
        selected_idx = np.random.choice(len(candidate_voxels), p=probs)
        selected_vox = candidate_voxels[selected_idx]
        if press_dom[selected_vox[0], selected_vox[1], selected_vox[2]] < X[2*un_t+term_node]:
            final_choice = candidate_voxels[selected_idx]
            satisfy = True
            
    
    #print("select_tissue_entry checks")
    
    return final_choice

# Core simulation modules
def simulate_arterial_vessel(current_position, X, a_element, adjacency_arterial, a_out_nodes, kappa_values ):
    path = []
    time_data = []
    
    while current_position in adjacency_arterial:
        next_nodes = [n for n in adjacency_arterial[current_position] 
                     if X[2*un_t + current_position] > X[2*un_t + n]]
        #print("next possible nodes: ", next_nodes)
        
            
        flows = np.array([kappa_values[(current_position, n)] * 
                         (X[2*un_t + current_position] - X[2*un_t + n])
                         for n in next_nodes])
        probs = flows / flows.sum()
        next_node = np.random.choice(next_nodes, p=probs)
        #print('chosen next node:', next_node)
        
        element = get_vessel_element(current_position, next_node, a_element)
        path.append((current_position, next_node))
        time_data.append(Time_art_vein(
            element['diameter'],
            X[2*un_t + current_position],
            X[2*un_t + next_node],
            element['length']
        ))
        #print(f'time for {(current_position, next_node)}:', time_data[-1])
        
        
        if next_node in a_out_nodes:
            #print('exits at:', next_node)
            break
        else:
            current_position = next_node
    

    #print("simulate_arterial_vessel checks")
        
    return next_node, path, time_data

def simulate_arterial_compartment(start_pos, press_dom_art, press_dom_ven, tissue_move_data):
    path = [start_pos]
    time_data = []
    current_pos = start_pos
    
    while tuple(current_pos) in tissue_move_data:
        valid_moves, flows = copy.deepcopy(tissue_move_data[tuple(current_pos)]) #deepcopy used so that it doesn't change the original
        X1 = press_dom_art[current_pos[0], current_pos[1], current_pos[2]]
        Xv = press_dom_ven[current_pos[0], current_pos[1], current_pos[2]]
        flow_vns = a*dV*(X1-Xv)
        
        if flow_vns > 0:
            valid_moves.append('venous')
            flows.append(flow_vns)
            
        probs = np.array(flows) / np.sum(flows)
        choice = np.random.choice(len(valid_moves), p=probs)
        
        if valid_moves[choice] == 'venous':
            #print('transitions to venous compartment at:', current_pos)
            break
            
        new_pos = valid_moves[choice]
        path.append(new_pos)
        time_data.append(Time_tissue(
            dLength(np.asarray(current_pos), np.asarray(new_pos)),
            X1,  
            press_dom_art[new_pos[0], new_pos[1], new_pos[2]],
            Ka
        ))
        current_pos = new_pos
    
    #print("simulate_arterial_compartment checks")    
    
    return current_pos, path, time_data

def simulate_venous_compartment(start_pos, press_dom_ven, nbr_dict, tissue_move_data, vflow, reversemap):
    path = [start_pos]
    time_data = []
    current_pos = tuple(start_pos)
    #print(current_pos)															 
    
    while current_pos in tissue_move_data.keys():
        X1 = press_dom_ven[current_pos[0], current_pos[1], current_pos[2]]
        #print(X1)
        valid_moves, flows = copy.deepcopy(tissue_move_data[current_pos]) #deepcopy used so that it doesn't change the original
        
        #print(f'for {current_pos}, valid moves: {valid_moves}, flows: {flows}')
        
        potential_v_out = []
        # current_pos_tuple = tuple(current_pos)
        if current_pos in reversemap:
            #outlets = nbr_dict[current_pos]
            node_flow_pairs = reversemap[current_pos]
            #print(node_flow_pairs)
            for (n,f) in node_flow_pairs:
                if X[2*un_t+un_a+n]<X1:
                    potential_v_out.append(n)
                    flows.append(f)
            #potential_v_out.extend(nbr_dict[current_pos])
            #flows.extend([f for (_, f) in reversemap[current_pos]])

        #print(f"potential v_outs for {current_pos}: {potential_v_out}")
            
        probs = np.array(flows) / np.sum(flows)
        #print(f'total possible next nodes to choose from:{len(probs)}')
        choice = np.random.choice(len(flows),size=1, p=probs)[0]
        #print (f"chosen idx: {choice}")
        
        if choice in range(len(valid_moves)):
            new_pos = valid_moves[choice]
            # print(new_pos)
            X2 = press_dom_ven[new_pos[0], new_pos[1], new_pos[2]]
            # print(X2)
            path.append(new_pos)
            
            
            time_data.append(Time_tissue(
                dLength(np.asarray(current_pos), np.asarray(new_pos)), X1,  X2, Kv))
                         
			
            current_pos = tuple(new_pos)
        else:
            current_pos = potential_v_out[choice-len(valid_moves)] ##indexing checked
            break
            
    #print("simulate_venous_compartment checks")        
        
    return current_pos, path, time_data

def simulate_venous_vessel(current_position, X, v_element, adjacency_venous, v_out_nodes, kappa_values):
    path = []
    time_data = []
    
    while current_position in adjacency_venous:
        
        next_nodes = [n for n in adjacency_venous[current_position] 
                     if X[2*un_t + un_a + current_position] > X[2*un_t + un_a + n]]
        #print("next possible nodes: ", next_nodes)
        if not next_nodes:
            break
            
        flows = np.array([kappa_values[(current_position, n)] * 
                         (X[2*un_t + un_a + current_position] - X[2*un_t + un_a + n])
                         for n in next_nodes])
        probs = flows / flows.sum()
        next_node = np.random.choice(next_nodes, p=probs)
        #print('chosen next node:', next_node)
        element = get_vessel_element(current_position, next_node, v_element)
        path.append((current_position, next_node))
        time_data.append(Time_art_vein(
            element['diameter'],
            X[2*un_t + un_a + current_position],
            X[2*un_t + un_a + next_node],
            element['length']
        ))
        #print(f'time for {(current_position, next_node)}:', time_data[-1])
        current_position = next_node
        
        if current_position in outlets:
            break
    

    #print("simulate_venous_vessel checks") 
        
    return current_position, path, time_data
    
def run_trial(args):
    (trial_idx, X, a_element, v_element, press_dom_a, press_dom_v, 
     adjacency_arterial, adjacency_venous, a_out_nodes, v_out_nodes,
     kappa_values_a, kappa_values_v, nbr_a_dict, nbr_v_dict, Ca, Cv, a_flow, v_flow, 
     reversemap_vox_vTerm, tissue_moves_art, tissue_moves_ven) = args 
    
    # seed = trial_idx  
    # np.random.seed(seed)

    try:
        print("trial_id:", trial_idx)
        # 1. Arterial Vessels
        start_node = np.random.choice(inlets)
        print("start node =", start_node)
        art_term_node, art_vessel_path, art_vessel_time = simulate_arterial_vessel(
            start_node, X, a_element, adjacency_arterial, a_out_nodes, kappa_values_a
        )
        
        print("term node =", art_term_node)
        
        
        tissue_entry = select_tissue_entry(art_term_node, nbr_a_dict, a_out_nodes, a_flow, X, press_dom_a)
        print('tissue entry at:', tissue_entry)
        
        
        venous_start, art_tissue_path, art_tissue_time = simulate_arterial_compartment(
            tissue_entry, press_dom_a, press_dom_v, tissue_moves_art
        )
        
        print('enters venous compartment at:', venous_start)
        
        
        ven_term_node, ven_tissue_path, ven_tissue_time = simulate_venous_compartment(
            venous_start, press_dom_v, nbr_v_dict, tissue_moves_ven, v_flow, reversemap_vox_vTerm
        )
        
        print('enters venous vesssel at:', ven_term_node)
        
        out, ven_vessel_path, ven_vessel_time = simulate_venous_vessel(ven_term_node, X, v_element, adjacency_venous, v_out_nodes, kappa_values_v)
        print('final outlet:', out)
        
        #return sum(art_vessel_time) + sum(art_tissue_time) + sum(ven_tissue_time) + sum(ven_vessel_time)
        return {
            'status': 'success',
            'total_transit_time': sum(art_vessel_time) + sum(art_tissue_time) + sum(ven_tissue_time) + sum(ven_vessel_time),
            'times': [art_vessel_time, art_tissue_time, ven_tissue_time, ven_vessel_time],
            'artery_path': art_vessel_path,
            'arterial_path': art_tissue_path,
            'venous_path': ven_tissue_path,
            'vein_path': ven_vessel_path
        }                   #'seed': seed,

        
    except Exception as ex:
        print(f"Trial {trial_idx} failed: {str(ex)}") #(seed={seed})
        # Save failure information
        failure_data = {
            'status': 'failed',
            'error': str(ex),
            'trial_idx': trial_idx,
        }    #'seed': seed,
        return failure_data    




def save_results_to_hdf5(all_trial_results, filename="transit_results.h5"):
    
    with h5py.File(filename, 'w') as f:
        # Store global metadata
        f.attrs['simulation_date'] = time.ctime()
        f.attrs['total_trials'] = len(all_trial_results)
        f.attrs['success_count'] = sum(1 for r in all_trial_results if r['status'] == 'success')
        f.attrs['failure_count'] = sum(1 for r in all_trial_results if r['status'] == 'failed')
        f.attrs['file_version'] = '1.6'
        
        # Store all trials
        for trial_idx, trial in enumerate(all_trial_results):
            trial_grp = f.create_group(f"trial_{trial_idx:06d}")
            trial_grp.attrs['status'] = trial['status']
            #trial_grp.attrs['seed'] = trial['seed']
            
            if trial['status'] == 'success':
                # Store each time component as separate variable-length dataset
                for i, component in enumerate(['artery', 'arterial_tissue', 
                                              'venous_tissue', 'vein']):
                    trial_grp.create_dataset(
                        f'times_{component}',
                        data=np.array(trial['times'][i], dtype='f8'),
                        compression="gzip" if len(trial['times'][i]) > 100 else None
                    )
                
                # Store paths
                for path in ['artery_path', 'arterial_path', 'venous_path', 'vein_path']:
                    if trial[path]:  # Only create if non-empty
                        dtype = 'i4' if 'artery' in path or 'vein' in path else 'f4'
                        trial_grp.create_dataset(
                            path,
                            data=np.array(trial[path], dtype=dtype),
                            compression="gzip" if len(trial[path]) > 100 else None
                        )
                
                # Store total time for convenience
                trial_grp.attrs['total_transit_time'] = sum(sum(t) for t in trial['times'])
            
            else:  # Failed trial
                trial_grp.attrs['error'] = trial['error']
                trial_grp.attrs['original_index'] = trial.get('trial_idx', trial_idx)
                
                # Create empty datasets for consistency
                for component in ['artery', 'arterial_tissue', 'venous_tissue', 'vein']:
                    trial_grp.create_dataset(
                        f'times_{component}',
                        data=np.empty(0, dtype='f8')
                    )
                
                for path, dtype in [('artery_path', 'i4'), ('arterial_path', 'f4'),
                                  ('venous_path', 'f4'), ('vein_path', 'i4')]:
                    trial_grp.create_dataset(path, data=np.empty(0, dtype=dtype))
                    

def save_failures(failures, filename="FINAL_failed_trials.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(failures, f)
        

if __name__ == '__main__':
    # Load and preprocess data
    dom = np.load('tongue_3D.npy')
    c_dom = np.load('c_dom.npy')
    #X = np.load('solutions/0.00/flow_solution_new_X0_6.4e-05.npy', allow_pickle=True)
    X = np.load('new_method_flow_rcd/0.01/flow_solution_new_X0_6.4e-05.npy', allow_pickle=True)
    
    
    # Initialize pressure domains
    press_dom_a, press_dom_v = initialize_pressure_domains(dom, c_dom, X)
    
    # Load element data
    a_element = pd.read_csv('arteries_element_database.csv',index_col=0, header=0)
    cols=['initial node', 'final node', 'diameter', 'length']
    a_element.columns=cols
    
    v_element = pd.read_csv('veins_element_database.csv',index_col=0, header=0)
    cols=['initial node', 'final node', 'diameter', 'length']
    v_element.columns=cols
    
    
    a_out = pd.read_csv('arteries_outlet_coordinates_3D_shifted.csv')
    cols=['Node', 'x', 'y', 'z']
    a_out.columns = cols
    a_out_nodes = {row['Node']: (row['x'], row['y'], row['z']) for _, row in a_out.iterrows()}
    
    v_out = pd.read_csv('veins_outlet_coordinates_3D_shifted.csv')
    cols=['Node', 'x', 'y', 'z']
    v_out.columns=cols
    v_out_nodes = {row['Node']: (row['x'], row['y'], row['z']) for _, row in v_out.iterrows()}
    
    nbr_a = np.load('nbrhd_matrices/0.01/nbrhd_3D_a_dx_dy_6.4e-05_e_0.01_new.npy', allow_pickle=True).tolist()
    nbr_a_dict = {row['Node']: nbr_a[i] for i, row in a_out.iterrows()}  
    
    nbr_v = np.load('nbrhd_matrices/0.01/nbrhd_3D_v_dx_dy_6.4e-05_e_0.01_new.npy', allow_pickle=True).tolist()
    nbr_v_dict = {row['Node']: nbr_v[i] for i, row in v_out.iterrows()}
    
    Ca = np.load('constants/Ca_3D_dx_dy_6.4e-05_e_0.01.npy')
    Cv = np.load('constants/Cv_3D_dx_dy_6.4e-05_e_0.01.npy')
    
    
    # Precompute lookup structures
    adjacency_arterial = make_adjacency_dict(a_element)
    adjacency_venous = make_adjacency_dict(v_element)
    
    kappa_values_a = kappa_values_calculator(a_element)
    kappa_values_v = kappa_values_calculator(v_element)
    
    nbr_v_dict = defaultdict(list)
    for outlet_idx, voxel_list in enumerate(nbr_v):
        for voxel in voxel_list:
            outlet = v_out['Node'].iloc[outlet_idx]
            nbr_v_dict[tuple(voxel)].append(outlet)
    #nbr_v_dict = dict(nbr_v_dict) ##converts to regular dictionary
    
    a_flow = precompute_flows_aTerm_to_Vox(a_out_nodes, nbr_a, a_element, kappa_values_a, X, Ca, e)
    v_flow, reversemap_vox_vTerm = precompute_flows_Vox_to_vTerm(v_out_nodes, nbr_v, v_element, kappa_values_v, X, Cv, e)
    
    tissue_moves_art = precompute_tissue_movement_vectorized(dom, press_dom_a, Ka)
    tissue_moves_ven = precompute_tissue_movement_vectorized(dom, press_dom_v, Kv)
    
    constant_params = (
        X, a_element, v_element, press_dom_a, press_dom_v,
        adjacency_arterial, adjacency_venous, a_out_nodes, v_out_nodes,
        kappa_values_a, kappa_values_v, nbr_a_dict, nbr_v_dict,
        Ca, Cv, a_flow, v_flow, reversemap_vox_vTerm,
        tissue_moves_art, tissue_moves_ven
    )
    
    
    # Run simulation
    N_trial = 100000 #5000 10000 #
    trial_args = [(i,) + constant_params for i in range(N_trial)]
    
    results = []
    failures = []
    # run without paralellization
    start_time = time.time()
    for i in range(N_trial):
        #results.append(run_trial(trial_args[i]))
        
        result = run_trial(trial_args[i])
        results.append(result)
        if result['status'] == 'failed':
            failures.append(result)
    
    if failures:
        failures_filename = f"FINAL_failed_trials_e_{e}_N_{N_trial}.pkl"
        save_failures(failures, failures_filename)
    
    
    end_time = time.time()
    print(f"All trials ended after: {end_time-start_time}")
    print(f"Successes: {N_trial-len(failures)}, Failures: {len(failures)}")
    
    #print(f'results \n {results}')
    
    # Save to HDF5 with metadata
    #hdf5_filename = f"TEST_transit_results_e_{e}_N_{N_trial}_{time.strftime('%Y%m%d_%H%M%S')}.h5"
    hdf5_filename = f"FINAL2_transit_results_e_{e}_N_{N_trial}_{time.strftime('%Y%m%d')}_{file_path}.h5"
    save_results_to_hdf5(results, hdf5_filename)