import numpy as np
import pandas as pd
from numba import njit
from collections import defaultdict 
import glob
import os
import time
import h5py


# Constants
dz = 0.333e-3  # m
dx = dy = 6.4e-5  # m
dV = dx * dy * dz
dz_sq = dz**2
dx_sq = dx**2

e = 0.01  # m
myu = 3e-3  # PaÂ·s
Ka = 1e-12  # mÂ²
Kv = 5e-10  # mÂ²
a = 1e-6  # perfusion coefficient
gamma_v = 1e-14  # mÂ²
delta = 0.5 * 7.5e-6 #m avg radius of wbc(6-9 *1e-6 m)


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

@njit(fastmath=True, cache=True)
def SoI(center, point):
    
    diff = point - center
    
    #print(point, center, diff)
    
    # Weight by voxel dimensions and sum
    return np.sqrt(diff[0]*diff[0] * dx_sq + 
                  diff[1]*diff[1]*dx_sq +  # Using dx_sq since dy = dx
                  diff[2]*diff[2] * dz_sq)


@njit(fastmath=True, cache=True)
def distance(center, point):
    
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
def eff_radius(length, flow, p1, p2):
    return ((8*myu*length*flow)/(np.pi*(p1-p2)))**0.25

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


def process_h5_file(input_file, a_flow, v_flow):
    """Process a single HDF5 file and add jump time attributes"""
    with h5py.File(input_file, 'a') as f:
        for trial_name in f:
            if not trial_name.startswith('trial_'):
                continue
                
            trial = f[trial_name]
            if trial.attrs.get('status') != 'success':
                continue
                
            # Process arterial jump time
            artery_path = trial['artery_path'][:]
            final_artery_node = artery_path[-1][1] if len(artery_path) > 0 else None
            final_artery_node_coord = [a_out_nodes[final_artery_node][1], 
                                     a_out_nodes[final_artery_node][0], 
                                     a_out_nodes[final_artery_node][2]]
            
            arterial_path = trial['arterial_path'][:].astype(int).tolist()
            first_arterial_entry = arterial_path[0] if len(arterial_path) > 0 else None
            first_arterial_entry_idx = nbr_a_dict[final_artery_node].index(first_arterial_entry)
            flow = a_flow[final_artery_node][first_arterial_entry_idx]
            l = distance(np.array(final_artery_node_coord), np.array(first_arterial_entry))
            p_term, p_vox = X[2*un_t + final_artery_node], press_dom_a[tuple(first_arterial_entry)]
            r = eff_radius(l, flow, p_term, p_vox)
            time_art_jump = (np.pi*r*r*l)/flow
            trial.attrs['time_art_jump'] = float(time_art_jump)

            # Process venous jump time
            vein_path = trial['vein_path'][:]
            first_vein_node = vein_path[0][0] if len(vein_path) > 0 else None
            first_vein_node_coord = [v_out_nodes[first_vein_node][1], 
                                   v_out_nodes[first_vein_node][0], 
                                   v_out_nodes[first_vein_node][2]]
            
            venous_path = trial['venous_path'][:].astype(int).tolist()
            last_venous_exit = venous_path[-1] if len(venous_path) > 0 else None
            last_venous_exit_idx = nbr_v_dict[first_vein_node].index(last_venous_exit)
            flow = v_flow[first_vein_node][last_venous_exit_idx]
            l = distance(np.array(first_vein_node_coord), np.array(last_venous_exit))
            p_term, p_vox = X[2*un_t + un_a + first_vein_node], press_dom_v[tuple(last_venous_exit)]
            r = eff_radius(l, flow, p_vox, p_term)
            time_ven_jump = (np.pi*r*r*l)/flow
            trial.attrs['time_ven_jump'] = float(time_ven_jump)

def process_all_files(file_pattern="FINAL2_transit_results*.h5"):
    """Process all files matching the pattern"""
    input_files = sorted(glob.glob(file_pattern))
    if not input_files:
        print("No files found matching pattern")
        return

    # Process each file
    for input_file in input_files:
        print(f"Processing {os.path.basename(input_file)}...")
        start_time = time.time()
        process_h5_file(input_file, a_flow, v_flow)
        print(f"Completed in {time.time()-start_time:.2f} seconds")
    
    print(f"\nProcessed {len(input_files)} files")

# [Keep all your existing initialization code...]

if __name__ == "__main__":
    # Initialize all required data structures
    dom = np.load('tongue_3D.npy')
    c_dom = np.load('c_dom.npy')
    X = np.load('new_method_flow_rcd/0.01/flow_solution_new_X0_6.4e-05.npy', allow_pickle=True)
    
    press_dom_a, press_dom_v = initialize_pressure_domains(dom, c_dom, X)
    
    # Load element data
    a_element = pd.read_csv('arteries_element_database.csv', index_col=0, header=0)
    a_element.columns = ['initial node', 'final node', 'diameter', 'length']
    
    v_element = pd.read_csv('veins_element_database.csv', index_col=0, header=0)
    v_element.columns = ['initial node', 'final node', 'diameter', 'length']
    
    # Load outlet data
    a_out = pd.read_csv('arteries_outlet_coordinates_3D_shifted.csv')
    a_out.columns = ['Node', 'x', 'y', 'z']
    a_out_nodes = {row['Node']: (row['x'], row['y'], row['z']) for _, row in a_out.iterrows()}
    
    v_out = pd.read_csv('veins_outlet_coordinates_3D_shifted.csv')
    v_out.columns = ['Node', 'x', 'y', 'z']
    v_out_nodes = {row['Node']: (row['x'], row['y'], row['z']) for _, row in v_out.iterrows()}
    
    # Load neighborhood data
    nbr_a = np.load('nbrhd_matrices/0.01/nbrhd_3D_a_dx_dy_6.4e-05_e_0.01_new.npy', allow_pickle=True).tolist()
    nbr_a_dict = {row['Node']: nbr_a[i] for i, row in a_out.iterrows()}
    
    nbr_v = np.load('nbrhd_matrices/0.01/nbrhd_3D_v_dx_dy_6.4e-05_e_0.01_new.npy', allow_pickle=True).tolist()
    nbr_v_dict = {row['Node']: nbr_v[i] for i, row in v_out.iterrows()}
    
    # Load constants
    Ca = np.load('constants/Ca_3D_dx_dy_6.4e-05_e_0.01.npy')
    Cv = np.load('constants/Cv_3D_dx_dy_6.4e-05_e_0.01.npy')
    
    # Precompute flows
    kappa_values_a = kappa_values_calculator(a_element)
    kappa_values_v = kappa_values_calculator(v_element)
    a_flow = precompute_flows_aTerm_to_Vox(a_out_nodes, nbr_a, a_element, kappa_values_a, X, Ca, e)
    v_flow, _ = precompute_flows_Vox_to_vTerm(v_out_nodes, nbr_v, v_element, kappa_values_v, X, Cv, e)
    
    # Process all files
    start_time = time.time()
    process_all_files("FINAL2_transit_results*.h5")
    print(f"Total processing time: {(time.time()-start_time)/60:.2f} minutes")