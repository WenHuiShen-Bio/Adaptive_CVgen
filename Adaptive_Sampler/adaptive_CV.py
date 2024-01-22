from __future__ import print_function
import sys
import os

import pyemma
from deeptime.clustering import KMeans
from deeptime.decomposition import TICA
from deeptime.markov.tools import estimation
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM
import mdtraj as md
import numpy as np
from datetime import datetime
import multiprocessing
import argparse
import pickle
import random
from collections import Counter
from prody import parseDCD, DCDFile # for construct DCD file

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)


class CVgenerator:
    # initialization
    def __init__(self, pdb_structure, windows_size='auto', steps='auto', indices='C'):
        self.struc = pdb_structure
        self.indices = indices
        struc = md.load_pdb(pdb_structure)
        elements_name = np.array([str(i)[-1] for i in struc.topology.atoms])
        if indices=='all':
            isnot_H = np.where(elements_name!='H')[0]
            is_H = np.where(elements_name=='H')[0]
            selected_id_lst = [isnot_H,is_H]
            n_element = struc.topology.n_atoms
        elif indices=='C':
            selected_id_lst = [np.where(elements_name=='C')[0]]
            n_element = struc.topology.select("name == 'C'").shape[0]
            
        self.selected_id_lst = selected_id_lst
        if windows_size=='auto' and steps=='auto':
            self.windows_size = np.arange(8,n_element,4)
            self.steps = np.round(self.windows_size/2).astype(int)-1
            if self.windows_size[-1]!=n_element:
                self.windows_size = np.append(self.windows_size,n_element)
                self.steps = np.append(self.steps,n_element)
        else:
            self.windows_size = windows_size
            self.steps = steps
        
        
    def generateCV(self):
        struc = md.load_pdb(self.struc)
        n_element = struc.topology.select("name == 'C'").shape[0]
        
        CV_indices = []
        CV_residues_index = []
        CV_struc_indices = []
        if len(self.selected_id_lst)==1:
            arrs_in = self.selected_id_lst[0]
            for size,step in zip(self.windows_size,self.steps):
                arrs_out = self.generate_sliding_windows(arrs_in,size,step)
                for ress in arrs_out:
                    sel_indices = struc.topology.select('index '+' '.join(str(i) for i in ress))
                    if not any(np.array_equal(sel_indices, arr) for arr in CV_struc_indices):
                        CV_struc_indices.append(sel_indices)
                    if not any(np.array_equal(sel_indices, arr) for arr in CV_indices):
                        CV_indices.append(sel_indices)
                        CV_residues_index.append(ress)
        elif len(self.selected_id_lst)==2:
            arrs_in = self.selected_id_lst[0]
            for size,step in zip(self.windows_size,self.steps):
                arrs_out = self.generate_sliding_windows(arrs_in,size,step)
                for ress in arrs_out:
                    sel_indices = struc.topology.select('index '+' '.join(str(i) for i in ress))
                    if not any(np.array_equal(sel_indices, arr) for arr in CV_struc_indices):
                        CV_struc_indices.append(sel_indices)
                    if not any(np.array_equal(sel_indices, arr) for arr in CV_indices):
                        CV_indices.append(sel_indices)
                        CV_residues_index.append(ress)
                        
            coarse_bead = []
            heavy_atom_ids = self.selected_id_lst[0]
            light_atom_ids = self.selected_id_lst[1]
            heavy_positions = struc.xyz[0][heavy_atom_ids]
            light_positions = struc.xyz[0][light_atom_ids]
            C_H_cutoff = 1.2
            for n,heavy_xyz in enumerate(heavy_positions):
                dis = np.sum((light_positions-heavy_xyz)**2,axis=1)
                active_H = light_atom_ids[np.where(dis<=C_H_cutoff)]
                if active_H.shape[0]>0:
                    coarse_bead.append(list(np.append(heavy_atom_ids[n],active_H)))
                
            sizes = np.arange(2,heavy_atom_ids.shape[0]+1)
            steps = np.round(sizes/2).astype(int)-1
            steps[steps==0]=1
            for size,step in zip(sizes,steps):
                arrs_out = self.generate_sliding_windows(np.arange(heavy_atom_ids.shape[0]),size,step)
                for arr in arrs_out:
                    ress = []
                    for i in arr:
                        ress += coarse_bead[i]
                    sel_indices = struc.topology.select('index '+' '.join(str(i) for i in ress))
                    if not any(np.array_equal(sel_indices, arr) for arr in CV_struc_indices):
                        CV_struc_indices.append(sel_indices)
                    if not any(np.array_equal(sel_indices, arr) for arr in CV_indices):
                        CV_indices.append(sel_indices)
                        CV_residues_index.append(ress)  
            

        if not os.path.isdir('CV_structures'):
            os.mkdir('CV_structures')
            for n,indice in enumerate(CV_struc_indices):
                CV_struc = struc.atom_slice(indice)
                CV_struc.save_pdb('./CV_structures/CV_structure_'+str(n)+'.pdb')    

        return CV_indices, CV_residues_index
    
    def generate_sliding_windows(self, arr_in, window_size, step):
        if len(arr_in) <= window_size:
            lst_out = [list(arr_in)]
        else:
            lst_out = []
            arr_length = len(arr_in)
            start = 0
            while start < arr_length:
                end = start + window_size
                # Wrap around to the beginning if the end exceeds the array length
                if end > arr_length:
                    window = np.concatenate((arr_in[start:], arr_in[:end - arr_length]))
                else:
                    window = np.array(arr_in[start:end])
                lst_out.append(window.tolist())
                start += step
            
        return lst_out
    
def compute_prior_theta(CV_residues_index):
    from scipy.optimize import minimize
    def objective_function(I, S):
        F = np.dot(I, S)
        differences = np.abs(F[:, None] - F)
        return np.max(differences)

    def constraint(I):
        return np.sum(I) - 1
    
    all_res = [item for sublist in CV_residues_index for item in sublist]
    unique_res = np.unique(all_res)
    S = np.zeros((len(CV_residues_index),unique_res.shape[0]))
    for n,residues_index in enumerate(CV_residues_index):
        indices = np.array([np.where(unique_res==i)[0] for i in residues_index]).reshape(-1,)
        S[n,indices] = 1
    # print('R=',R)
    n = len(CV_residues_index)
    # Initial guess for theta
    initial_I = np.ones((n)) / n
    # Define the optimization problem
    delt = 0.8
    bounds = [((1/n)*(1-delt), (1/n)*(1+delt))] * n  # Bounds for I, [0, 1)
    constraints = [{'type': 'eq', 'fun': constraint}]  # Equality constraint: sum(I) = 1
    res = minimize(objective_function,
                   initial_I, args=(S,),
                   method='SLSQP',
                   bounds=bounds,
                   constraints=constraints)
    # res = minimize(objective_function, initial_I, args=(R,), method='CG')
    
    # Extract the optimized I
    optimized_I = res.x
    
    print("Optimized I:", optimized_I)
    print("Objective function value (Q):", res.fun)
    
    return optimized_I

def directional_relative_entropy(old_data, new_data):
    combined_data = np.concatenate((old_data, new_data))
    
    if np.mean(old_data) > np.mean(new_data):
        bias = 1
    else:
        bias = -1
    # Determine the common bin edges based on the combined data
    bins = np.histogram_bin_edges(combined_data, bins='auto')
    bin_width = bins[1]-bins[0]

    # Calculate the distributions of the old data and the data after adding the new data
    old_distribution, _ = np.histogram(old_data, bins=bins, density=True)
    new_distribution, _ = np.histogram(combined_data, bins=bins, density=True)

    # Calculate the relative entropy between the old distribution and the new distribution
    mask = old_distribution > 0  # Mask to exclude zero probabilities
    relative_entropy = np.sum(old_distribution[mask] * bin_width * np.log2(old_distribution[mask] / new_distribution[mask]))
    
    newarea = combined_data[combined_data<np.min(old_data)]
    if newarea.shape[0]>0:
        cutid1 = np.max(newarea)
        mask1 = bins<cutid1
        relative_entropy += np.sum(new_distribution[mask1])
    newarea = combined_data[combined_data>np.max(old_data)]
    if newarea.shape[0]>0:
        cutid2 = np.min(newarea)
        mask2 = bins>cutid2
        relative_entropy -= np.sum(new_distribution[mask2])

    return bias*relative_entropy

def min_relative_std(x,interval):
    arg_x = np.argmin(x)
    nth_traj = arg_x//interval
    seg_data = x[nth_traj*interval:(nth_traj+1)*interval]
    std = np.std(seg_data)
    relative_std = std/(np.max(x)-np.min(x))
    return relative_std

def cluster(traj_xyz,n_clusters=400):
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,random_state=0,batch_size=2048,n_init="auto",max_iter=200)
    first_dim = traj_xyz.shape[0]
    trans_data = traj_xyz.reshape(first_dim,-1)
    kmeans = kmeans.partial_fit(trans_data)
    cluster_labels = kmeans.labels_
    return cluster_labels

# our TS-GP optimizer
class ThompsonSamplingGP:
    
    # initialization
    def __init__(self, n_samples, x_bounds, X, y, interval_resolution=1000):
        
        # number of samples draw
        self.n_samples = n_samples
        
        # the bounds tell us the interval of x we can work
        self.bounds = x_bounds
        
        # interval resolution is defined as how many points we will use to 
        # represent the posterior sample
        # we also define the x grid
        self.interval_resolution = interval_resolution
        self.X_grid = np.linspace(self.bounds[0], self.bounds[1], self.interval_resolution)
        
        # history actions positions
        # print(X)
        # print(y)
        # self.X = np.array(X)
        # self.y = np.array(y)
        
    # fitting process
    def fit(self, X, y):
        
        # let us use the Matern kernel
        K = 0.1 * Matern(length_scale=0.1, length_scale_bounds=(1*1e-0, 10.0), nu=1.5)

        # instance of GP
        gp = GaussianProcessRegressor(kernel=K)

        # fitting the GP
        gp.fit(X, y)
        
        # return the fitted model
        return gp
    
    # process of choosing next point
    def choose_next_samples(self):

        # 1. Fit the GP to the observations we have
        self.gp = self.fit(self.X.reshape(-1,1), self.y)
        
        # 2. Draw samples (a function) from the posterior
        posterior_samples = self.gp.sample_y(self.X_grid.reshape(-1,1), self.n_samples)
        
        # 3. Choose next point as the optimum of the sample
        which_min = np.array([])
        next_samples_y = np.array([])
        for posterior_sample in posterior_samples.T:
            which_min = np.append(which_min, np.argmin(posterior_sample))
            next_samples_y = np.append(next_samples_y, np.min(posterior_sample))
        which_min = which_min.astype(int)
        next_samples_x = self.X_grid[which_min]
        
        # # let us also get the std from the posterior, for visualization purposes
        # posterior_mean, posterior_std = self.gp.predict(self.X_grid.reshape(-1,1), return_std=True)
 
        return next_samples_x, next_samples_y
    
def random_select_indices(arr, N):
    indices = np.arange(len(arr))
    np.random.shuffle(indices)
    sel_indices = indices[:N]
    sel_values = arr[sel_indices]
    return sel_values, sel_indices

def choose_action(arms_values, arms_indices, N_per_arm, N_tot_actions, max_score=True):
    # list of samples, for each bandit
    samples_list = []
    indices_list = []
                
    # drawing a sample from each bandit distribution
    for values, values_indices in zip(arms_values,arms_indices):
        if values.shape[0]==0:
            continue
        sampled_values, indices = random_select_indices(values, N_per_arm)
        samples_list += list(sampled_values)
        indices_list += list(values_indices[indices])
    samples_list = np.array([samples_list][0])
    indices_list = np.array(indices_list)
    # print(samples_list)
    # print(indices_list)
    
    if max_score:  
        index = np.argsort(samples_list)[-N_tot_actions:]
    else:
        index = np.argsort(samples_list)[:N_tot_actions]
    
    sampled_indices = list(indices_list[index])
    # print(sampled_indices)
    return sampled_indices
    
    
def thompson_sampling_MSM(lagtime,n_clusters,bio_feat=np.array([]),prior_score=np.array([]),prior_score_indices=np.array([])):

    if(os.path.isfile('data_output.pkl')):
        with open('data_output.pkl','rb') as in_data:
            data_output = pickle.load(in_data)

    if bio_feat.shape[0] != 0:
        tics = bio_feat
    else:
        tica_estimator = TICA(lagtime=lagtime, dim=2)
        tica = tica_estimator.fit(data_output).fetch_model()
        print('TICA dimension = ', tica.output_dimension)
        tics = tica.transform(data_output)

    tmp_n_clusters = n_clusters
    # if(os.path.isfile('parameters.pkl')):
    #     with open('parameters.pkl','rb') as r:
    #         tmp_n_clusters = pickle.load(r)
    #     tmp_n_clusters += 10
    while True:
        try:
            cluster = KMeans(tmp_n_clusters, max_iter=200).fit_fetch(np.concatenate(tics)[::10])
            dtrajs = [cluster.transform(traj) for traj in tics]
            dtrajs_concatenated = np.concatenate(dtrajs)
            
            counts_estimator = TransitionCountEstimator(lagtime=lagtime, count_mode='sliding')
            counts = counts_estimator.fit_fetch(dtrajs).submodel_largest()
            msm_estimator = MaximumLikelihoodMSM()
            msm = msm_estimator.fit_fetch(counts)
            stat_dis = msm.stationary_distribution
            if len(stat_dis) != tmp_n_clusters:
                tmp_n_clusters = len(stat_dis)
                continue
            else:
                break
        except Exception as e:
            tmp_n_clusters -= 3
            # print("Error occurred:", str(e))
    print("Exception handling, tmp_n_clusters=",tmp_n_clusters)
    print('dtraj index start from ', np.min(dtrajs))
    
        
    arms_values = []
    arms_indices = []
    if prior_score.shape[0] != 0:
        for i in range(tmp_n_clusters):
            indices = np.where(dtrajs_concatenated==i)[0]
            inter_indices = np.intersect1d(indices,prior_score_indices)
            if inter_indices.shape[0]==0:
                arms_values.append(np.array([]))
                arms_indices.append(np.array([]))
            else:
                index = np.where(prior_score_indices == inter_indices[:, None])[1]
                arms_values.append(prior_score[index])
                arms_indices.append(inter_indices)
    else:
        d_awards = np.array([np.max(stat_dis[np.unique(dtraj)]) for dtraj in dtrajs])
        actions_in_arms = np.array([dtraj[0] for dtraj in dtrajs])        

        for i in range(tmp_n_clusters):
            id_lst = np.where(actions_in_arms==i)[0]
            if id_lst.shape[0]==0:
                arms_values.append(np.arange(0,int(np.max(stat_dis)*1000))/1000)
            else:
                arms_values.append(d_awards[id_lst])
    
    tran_matrix = msm.transition_matrix
    
    if not os.path.isdir('./TSplot'):
        os.mkdir('./TSplot')
    with open('./TSplot/tics.pkl','ab') as w:
        pickle.dump(tics,w)
    with open('./TSplot/dtrajs.pkl','ab') as w:
        pickle.dump(dtrajs,w)
    with open('./TSplot/msm.pkl','ab') as w:
        pickle.dump(msm,w)
    with open('./TSplot/stationary_distribution.pkl','ab') as w:
        pickle.dump(stat_dis,w)
    with open('./TSplot/transition_matrix.pkl','ab') as w:
        pickle.dump(tran_matrix,w)

    if prior_score.shape[0] != 0:
        return arms_values, arms_indices
    else:
        return arms_values, dtrajs

def cal_relative_fluctuation(data):
    return np.std(data)/np.mean(data)

def best_hummer_q(args):
    """Compute the fraction of native contacts according the definition from
    Best, Hummer and Eaton [1]
    
    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    native : md.Trajectory
        The 'native state'. This can be an entire trajecory, or just a single frame.
        Only the first conformation is used
        
    Returns
    -------
    q : np.array, shape=(len(traj),)
        The fraction of native contacts in each frame of `traj`
        
    References
    ----------
    ..[1] Best, Hummer, and Eaton, "Native contacts determine protein folding
          mechanisms in atomistic simulations" PNAS (2013)
    """
    traj, native = args
    import warnings
    import mdtraj as md
    import numpy as np
    from itertools import combinations

    BETA_CONST = 50  # 1/nm
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45  # nanometers
    
    # get the indices of all of the heavy atoms
    heavy = native.topology.select_atom_indices('heavy')
    # get the pairs of heavy atoms which are farther than 3
    # residues apart
    heavy_pairs = np.array(
        [(i,j) for (i,j) in combinations(heavy, 2)
            if abs(native.topology.atom(i).residue.index - \
                   native.topology.atom(j).residue.index) > 3])
    
    # compute the distances between these pairs in the native state
    heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
    # and get the pairs s.t. the distance is less than NATIVE_CUTOFF
    native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]
    # print("Number of native contacts", len(native_contacts))
    
    # now compute these distances for the whole trajectory
    r = md.compute_distances(traj, native_contacts)
    # and recompute them for just the native state
    r0 = md.compute_distances(native[0], native_contacts)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q

def low_count_MSM(lagtime,n_clusters):
    """identify the least count cluster from MSM

    Args:
        lagtime (_type_): _description_
        n_clusters (_type_): _description_

    Returns:
        _type_: _description_
    """
    if(os.path.isfile('data_output.pkl')):
        with open('data_output.pkl','rb') as in_data:
            data_output = pickle.load(in_data)

    tica_estimator = TICA(lagtime=lagtime, dim=2)
    tica = tica_estimator.fit(data_output).fetch_model()
    print('TICA dimension = ', tica.output_dimension)
    tics = tica.transform(data_output)

    cluster = KMeans(n_clusters, max_iter=200).fit_fetch(np.concatenate(tics)[::10])
    dtrajs = cluster.transform(np.concatenate(tics))
    counts_estimator = TransitionCountEstimator(lagtime=lagtime, count_mode='sliding')
    counts = counts_estimator.fit_fetch(dtrajs).submodel_largest()
    msm_estimator = MaximumLikelihoodMSM()
    msm = msm_estimator.fit_fetch(counts)
    stat_dis = msm.stationary_distribution
    tran_matrix = msm.transition_matrix
    with open('./stationary_distribution.pkl','ab') as w:
        pickle.dump(stat_dis,w)
    with open('./transition_matrix.pkl','ab') as w:
        pickle.dump(tran_matrix,w)

    c_matrix = estimation.count_matrix(dtrajs, lag=lagtime)
    transition_count = c_matrix.sum(axis=0)
    # diagonal = np.array([c_matrix[i, i] for i in range(num_cluster)])
    # cluster_score = diagonal/np.max(diagonal)
    cluster_score = transition_count/np.max(transition_count)
    cluster_score = cluster_score.getA()[0]
    state_score = np.array([cluster_score[i] for i in dtrajs])

    return state_score, dtrajs

def feature(dirs, pdb, interval):
    # feat = pyemma.coordinates.featurizer(pdb)
    # pairs = feat.pairs(feat.select('all'))
    # feat.add_contacts(pairs,threshold=0.8)
    # data = pyemma.coordinates.source(dirs, features=feat)
    orginal_data = pyemma.coordinates.source(dirs,top=pdb)
    data = orginal_data.get_output()
    # print(data)
    data_output = []
    inter = interval
    for d in data:
        n = d.shape[0]//inter
        for l in range(n):
            data_output.append(d[l*inter:(l+1)*inter])
    # print(data_output)
    return data_output

def cal_rest_index(traj, rmsd, cutoff, orin_index):
    rest_index = []
    for n,i in enumerate(rmsd):
        if i>cutoff:
            rest_index.append(n)

    max_rmsd = np.max(rmsd)
    index = np.where(rmsd==max_rmsd)
    ref_traj = traj[index]
    index = orin_index[index[0][0]]
    return rest_index, ref_traj, index

def sample_states(whole_traj,reference,num_states,n_iter):
    import copy
    heavy = reference.top.select_atom_indices('heavy')
    rmsd = md.rmsd(whole_traj,reference,frame=0,atom_indices=heavy)
    ind = np.where(rmsd==np.min(rmsd))
    init_ref = whole_traj[ind]

    cutoff = 0.8
    r_cutoff = 10
    l_cutoff = 0
    error = 100
    num_iterations = 0
    heavy = init_ref.top.select_atom_indices('heavy')
    n_times = 0
    old_left = 0
    while error>0:
        num_iterations += 1
        if num_iterations>n_iter:
            break
        selected_states = [init_ref]
        selected_indexs = [ind[0][0]]
        orin_index = np.array(range(len(whole_traj)))
        traj = copy.deepcopy(whole_traj)
        for iter in range(num_states-1):
            if iter==0:
                ref=init_ref
            rmsd = md.rmsd(traj,ref,atom_indices=heavy)
            rest_index, ref, ref_ind = cal_rest_index(traj,rmsd,cutoff,orin_index)
            orin_index = orin_index[rest_index]
            selected_states.append(ref)
            selected_indexs.append(ref_ind)
            xyz = traj.xyz[rest_index]
            traj.xyz = xyz
            # print(len(traj),' frames left')
            if len(traj)==0 and iter<num_states-1:
                r_cutoff = cutoff
                cutoff = (l_cutoff+r_cutoff)/2
                break
            if iter==num_states-1:
                error = len(traj)
                if np.max(rmsd)>cutoff:
                    l_cutoff = cutoff
                    cutoff = (l_cutoff+r_cutoff)/2
                else:
                    r_cutoff = cutoff
                    cutoff = (l_cutoff+r_cutoff)/2
        if n_times > 3:
            break
        if len(traj) == old_left:
            n_times += 1
        old_left = len(traj)
    return selected_states, selected_indexs

def cal_rmsd_score(args):
    """compute rmsd score

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    traj, ref, heavy = args
    rmsd = md.rmsd(traj, reference=ref, atom_indices=heavy)

    return rmsd

def cal_mab_score(args):
    """compute max-adjacency-bonds score
        C60 property:
            Number of nearest neighbor atoms : 3 (cutoff<1.8 unit-angstrom)
            Number of next nearest neighbor atoms : 6 (1.8<cutoff<2.6 unit-angstrom)
            diameter : 7.1 unit-angstrom
            radius : 3.55 unit-angstrom
            average cumulative distance : 29 unit-angstrom

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    scaling_factor = 50
    diameter = 0.71
    mean_cumulant_dis = 2.9 # cumulant distance for one bead
    first_cutoff = 0.18
    first_neighbors = 3
    second_cutoff = 0.26
    second_neighbors = 6
    
    import itertools
    traj, pdb, indices = args
    ref = md.load_pdb(pdb)
    n_atoms = ref.top.n_atoms
    all_atoms = ref.top.select_atom_indices('all')
    COM = md.compute_center_of_mass(ref)
    target_radius = traj.xyz-COM
    target_radius_value = diameter/2
    
    # print('pairs_distances.shape=',pairs_distances.shape)
    # print(np.max(pairs_distances,axis=1))
    # print('n_atoms=',n_atoms)
        
    cache_score = []
    for indice in indices:
        pairs = np.array(list(itertools.product(indice, np.array(all_atoms))))#.reshape(len(selected_atoms),len(n_atoms),-1)
        pairs_distances = md.compute_distances(traj,pairs)
        selected_atoms = indice
        score = []
        # for i in range(selected_atoms.shape[0]):
        #     start = i*n_atoms
        #     end = start+n_atoms
        #     single_pair_dis = pairs_distances[:,start:end]
        #     max_bonds_score = np.abs(np.sum((single_pair_dis<cutoff)&(single_pair_dis!=0),axis=1)-max_bonds).reshape(-1,)
        #     print('max_bonds_score=',max_bonds_score)
        #     # diameter_score = np.abs(np.max(single_pair_dis,axis=1)-diameter).reshape(-1,)
        #     # print('diameter_score=',diameter_score)
        #     cumulant_dis_score = np.abs(np.sum(single_pair_dis,axis=1)-mean_cumulant_dis).reshape(-1,)
        #     print('cumulant_dis_score=',cumulant_dis_score)
        #     # cache_score.append(max_bonds_score+diameter_score+cumulant_dis_score)
        #     score.append(max_bonds_score+cumulant_dis_score)
        # cache_score.append(np.mean(score,axis=0))

        selected_atoms_reshaped = selected_atoms.reshape(-1, 1)
        pairs_distances_reshaped = pairs_distances.reshape(len(traj), -1, n_atoms)

        first_neighbors_mask = np.logical_and(pairs_distances_reshaped < first_cutoff, pairs_distances_reshaped != 0)
        first_neighbors_score = np.abs(np.sum(first_neighbors_mask, axis=2) - first_neighbors).reshape(len(traj),-1)
        first_neighbors_score = np.mean(first_neighbors_score,axis=1)
        # print('first_neighbors_score=', first_neighbors_score)
        
        second_neighbors_mask = np.logical_and(pairs_distances_reshaped > first_cutoff, pairs_distances_reshaped < second_cutoff)
        second_neighbors_score = np.abs(np.sum(second_neighbors_mask, axis=2) - second_neighbors).reshape(len(traj),-1)
        second_neighbors_score = np.mean(second_neighbors_score,axis=1)
        # print('second_neighbors_score=', second_neighbors_score)

        cumulant_dis_score = np.abs(np.sum(pairs_distances_reshaped, axis=2) - mean_cumulant_dis).reshape(len(traj),-1)
        cumulant_dis_score = np.mean(cumulant_dis_score,axis=1)
        # print('cumulant_dis_score=', cumulant_dis_score)
        
        radius = target_radius[:,indice,:]
        radius_score = np.abs(np.mean(np.sqrt(np.sum(radius**2,axis=2)),axis=1)-target_radius_value)

        score = scaling_factor*(first_neighbors_score+second_neighbors_score) + cumulant_dis_score + scaling_factor*radius_score
        cache_score.append(score)
        
    mab_scores= np.array(cache_score)

    return mab_scores

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
#                For contact analysis
def filter_contact_include(ref_contact, filter_arr):
    true_contact = ref_contact.tolist()
    for pair in ref_contact:
        if sum(np.isin(filter_arr, pair[0])) and sum(np.isin(filter_arr, pair[1])):
            continue
        else:
            true_contact.remove(pair.tolist())
    true_contact = np.array(true_contact)
    return true_contact

def filter_contact_exclude(ref_contact, filter_arr):
    true_contact = ref_contact.tolist()
    for pair in ref_contact:
        if sum(np.isin(filter_arr, pair[0])) and sum(np.isin(filter_arr, pair[1])):
            true_contact.remove(pair.tolist())
    true_contact = np.array(true_contact)
    return true_contact

def analyze_contact(pdb):
    ref_contact = md.compute_contacts(pdb,scheme='closest-heavy')
    ref_dssp_code = md.compute_dssp(pdb, simplified=False)
    
    # --------------------------------------------------------------------------------------------------
    # analyze alpha-helix and beta-sheet
    flag_alpha = False
    flag_beta = False
    alpha_res_index = np.where(ref_dssp_code[0]=='H')
    if alpha_res_index[0].shape[0]>0:
        alpha_res = extract_consecutive_parts(alpha_res_index[0])
        flag_alpha = True
    beta_res_index = np.where(ref_dssp_code[0]=='E')
    if beta_res_index[0].shape[0]>0:
        beta_res = extract_consecutive_parts(beta_res_index[0])
        flag_beta = True
    
    if flag_alpha and flag_beta:
        stru_res = np.concatenate(alpha_res+beta_res)
    elif flag_alpha and (not flag_beta):
        stru_res = np.concatenate(alpha_res)
    elif (not flag_alpha) and flag_beta:
        stru_res = np.concatenate(beta_res)
    
    # --------------------------------------------------------------------------------------------------
    # first filter: include structure res and exclude inter-res contact within structures
    true_contact_pair = filter_contact_include(ref_contact[1],stru_res)
    if flag_alpha:
        for i_alpha in alpha_res:
            true_contact_pair = filter_contact_exclude(true_contact_pair,i_alpha)
    if flag_beta:
        true_contact_pair = filter_contact_exclude(true_contact_pair,np.concatenate(beta_res))
    ref_contact = md.compute_contacts(pdb, contacts=true_contact_pair, scheme='closest-heavy')
    
    short_contact_pair = []
    long_contact_pair = []
    for n,dis in enumerate(ref_contact[0][0]):
        if dis<=1.2:
            short_contact_pair.append(np.array(np.append(ref_contact[1][n],n)))
            # print('short contact :',ref_contact[1][n],'=',ref_contact[0][0][n])
        else:
            long_contact_pair.append(np.array(np.append(ref_contact[1][n],n)))
            # print('long contact :',ref_contact[1][n],'=',ref_contact[0][0][n])
    short_contact_pair = np.array(short_contact_pair)
    long_contact_pair = np.array(long_contact_pair)
    
    # --------------------------------------------------------------------------------------------------
    # second filter: identify the key residure pair
    true_short_contact_pair = []
    for res in np.sort(stru_res):
        ind = np.isin(short_contact_pair[:,:2],res)[:,0]+np.isin(short_contact_pair[:,:2],res)[:,1]
        ind = np.where(ind)[0]
        if ind.shape[0]==0:
            continue
        ind = np.array(short_contact_pair[:,2])[ind]
        dis_lst = ref_contact[0][0][ind]
        id = np.where(dis_lst == np.min(dis_lst))[0]
        true_short_contact_pair.append(ref_contact[1][ind[id]][0])
    true_short_contact_pair = np.array(true_short_contact_pair)

    true_long_contact_pair = []
    for res in np.sort(stru_res):
        ind = np.isin(long_contact_pair[:,:2],res)[:,0]+np.isin(long_contact_pair[:,:2],res)[:,1]
        ind = np.where(ind)[0]
        if ind.shape[0]==0:
            continue
        ind = np.array(long_contact_pair[:,2])[ind]
        dis_lst = ref_contact[0][0][ind]
        id = np.where(dis_lst == np.max(dis_lst))[0]
        true_long_contact_pair.append(ref_contact[1][ind[id]][0])
    true_long_contact_pair = np.array(true_long_contact_pair)
    # --------------------------------------------------------------------------------------------------
    
    ref_contact_short = md.compute_contacts(pdb, contacts=np.unique(true_short_contact_pair,axis=0), scheme='closest-heavy')
    ref_contact_long = md.compute_contacts(pdb, contacts=np.unique(true_long_contact_pair,axis=0), scheme='closest-heavy')
    
    return ref_contact_short, ref_contact_long

def contact_score(contacts, ref_contact, error):
    score_matrix = np.absolute(contacts[0] - ref_contact[0])
    N = score_matrix.shape[1]
    count_matrix = np.sum(score_matrix < error, axis=1)
    scores = 1 - count_matrix / N
    return scores
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

def cal_contact_score(args):
    """compute the contact score

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    traj_slice,pdb = args
    ref_contact_short, ref_contact_long = analyze_contact(pdb)
    contacts_short = md.compute_contacts(traj_slice, contacts=ref_contact_short[1], scheme='closest-heavy')
    contacts_long = md.compute_contacts(traj_slice, contacts=ref_contact_long[1], scheme='closest-heavy')
    scores_short = contact_score(contacts_short,ref_contact_short,0.05)
    scores_long = contact_score(contacts_long,ref_contact_long,0.05)
    return scores_short, scores_long
    
    
def cal_dssp_score(args):
    """compute sencondary structure score

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """

    traj_slice,ref_code,beta_lst,prior_b_pair,minor_b_pair = args
    ## This calculation not include random coil structures

    # calculate the percentage of matching elements
    # helix
    ntot_res = ref_code[0].shape[0]
    h_index = np.where(ref_code[0]=='H')
    n_alpha_res = h_index[0].shape[0]
    if prior_b_pair is None:
        n_prior_beta_res=0
    else:
        prior_set = set()
        for pair in prior_b_pair:
            prior_set |= set(pair)
        n_prior_beta_res = 0
        for i in prior_set:
            n_prior_beta_res += beta_lst[i].shape[0]
    if minor_b_pair is None:
        n_minor_beta_res=0
    else:
        minor_set = set()
        for pair in minor_b_pair:
            minor_set |= set(pair)
        n_minor_beta_res = 0
        for i in minor_set:
            n_minor_beta_res += beta_lst[i].shape[0]
    dssp_code = md.compute_dssp(traj_slice,simplified=True)
    award_score = []
    # punish_score = []
    for code in dssp_code:
        award_score.append(np.sum((ref_code[0]==code) & (code!='E'))/ntot_res)
        # punish_score.append(np.sum((ref_code[0]!=code) & (code!=' '))/ntot_res)
    award_score = np.array(award_score)
    # punish_score = np.array(punish_score)
    # helix_score = 1-award_score+punish_score
    helix_score = 1-award_score


    # beta-sheet
    sheet_1_score = None
    sheet_2_score = None
    if sum(np.isin(ref_code[0], 'E')):
        prior_beta_score = []
        minor_beta_score = []
        for code in dssp_code:
            # prior
            score = 0
            for p in prior_b_pair:
                beta1 = beta_lst[p[0]]
                beta2 = beta_lst[p[1]]
                n1 = sum(np.isin(code[beta1[0]:beta1[-1]+1], 'E'))
                n2 = sum(np.isin(code[beta2[0]:beta2[-1]+1], 'E'))
                score += (n1+n2)/ntot_res
            prior_beta_score.append(score)
            # minor
            score = 0
            for p in minor_b_pair:
                beta1 = beta_lst[p[0]]
                beta2 = beta_lst[p[1]]
                n1 = sum(np.isin(code[beta1[0]:beta1[-1]+1], 'E'))
                n2 = sum(np.isin(code[beta2[0]:beta2[-1]+1], 'E'))
                score += (n1+n2)/(n_minor_beta_res)
            minor_beta_score.append(score)

        prior_beta_score = np.array(prior_beta_score)
        sheet_1_score = 1-prior_beta_score
        minor_beta_score = np.array(minor_beta_score)
        sheet_2_score = 1-(minor_beta_score)

    return helix_score, sheet_1_score, sheet_2_score

def extract_consecutive_parts(arr):
    mask = np.insert(np.diff(arr) != 1, 0, True)
    indices = np.where(mask)[0]

    # Check if the first element is part of a consecutive sequence
    if arr[0] - 1 not in arr:
        indices = indices[1:]

    consecutive_parts = np.split(arr, indices)
    return consecutive_parts

def cal_AF_score(args):
    """compute the alphafold weight score

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    traj, pairs, bins, probs2, predicted_prob = args
    bin_width = (bins[-1]-bins[0])/63
    pairs_dis = md.compute_distances(traj,pairs)
    indices = np.array(np.ceil((pairs_dis-bins[0])/bin_width),dtype=int)
    indices[np.where(indices<0)]=0
    indices[np.where(indices>63)]=63
    AF_scores = []
    for indice in indices:
        n = 0
        prob = 0
        for i in range(probs2.shape[0]-1):
            for j in range(i+1,probs2.shape[0]):
                prob += probs2[i,j,indice[n]]
                n += 1
        prob = 1-(prob/predicted_prob)
        AF_scores.append(prob)
    AF_scores = np.array(AF_scores)
    return AF_scores

def cal_U_score(args):
    """compute the potential energy

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    import openmm.unit as u
    import openmm.app as app
    import openmm as mm
    traj, pdb_name = args
    pdb = app.PDBFile(pdb_name)
    forcefield = app.ForceField('amber14/protein.ff14SB.xml', 'implicit/gbn2.xml')
    # # Create the OpenMM system
    # print('Creating OpenMM System')
    system = forcefield.createSystem(pdb.topology, 
                                        nonbondedMethod=app.NoCutoff,
                                    constraints=app.HBonds
                                )
    integrator = mm.LangevinIntegrator(
                                300*u.kelvin,       # Temperature of heat bath
                                1.0/u.picosecond,  # Friction coefficient
                                0.0025*u.picoseconds  # Time step
        )
    topology = pdb.topology
    sim = app.Simulation(topology, system, integrator)
    energy = []
    for t in traj:
        positions = t.xyz[0]
        sim.context.setPositions(positions)
        state = sim.context.getState(getEnergy=True)
        energy.append(state.getPotentialEnergy())
    U=[]
    for e in energy:
        U.append(e.value_in_unit(u.kilojoule/u.mole))
    U = np.array(U)
    return U

    
def call_cal_struc_score(traj_slices, metrix='SS'):
    """compute global CV score

    Args:
        traj_slices (list): a list of md.traj
        metrix (str, optional): _description_. Defaults to 'SS'.

    Returns:
        _type_: _description_
    """
    pdb = args.pdb
    ref = md.load_pdb(pdb)
    heavy = ref.top.select("all")

    N = multiprocessing.cpu_count()
  
    if metrix=='R':
        # calculate rmsd score
        pool = multiprocessing.Pool(processes=N)
        rmsd_results = pool.map(cal_rmsd_score, [(traj_slice, ref, heavy) for traj_slice in traj_slices])
        pool.close()
        pool.join()
        rmsd_score = np.concatenate(rmsd_results)
        metrix_score = rmsd_score
    
    if metrix=='Q':
        # calculate native contact best_hummer_q
        pool = multiprocessing.Pool(processes=N)
        Q_results = pool.map(best_hummer_q, [(traj_slice, ref) for traj_slice in traj_slices])
        pool.close()
        pool.join()
        Q_score = np.concatenate(Q_results)
        metrix_score = 1 - Q_score
        
    if metrix=='AF':
        # calculate predictive score from alphafold
        if args.alphafold_probs is None:
            parser.error('Need to specify alphafold_probs file ')
        with open(args.alphafold_probs,'rb') as r:
            pairs = pickle.load(r)
            bins = pickle.load(r)
            probs2 = pickle.load(r)
            predicted_prob = pickle.load(r)
        pool = multiprocessing.Pool(processes=N)
        AF_results = pool.map(cal_AF_score, [(traj_slice, pairs, bins, probs2, predicted_prob) for traj_slice in traj_slices])
        pool.close()
        pool.join()
        AF_score = np.concatenate(AF_results)
        metrix_score = AF_score
        
    if metrix=='U':
        # calculate potential energy
        pool = multiprocessing.Pool(processes=N)
        U_results = pool.map(cal_U_score, [(traj_slice, pdb) for traj_slice in traj_slices])
        pool.close()
        pool.join()
        U_score = np.concatenate(U_results)
        metrix_score = U_score
    
    if metrix=='contact':
        # calculate short and long contact scores
        pool = multiprocessing.Pool(processes=N)
        contact_results = pool.map(cal_contact_score, [(traj_slice, ref) for traj_slice in traj_slices])
        pool.close()
        pool.join()
        scores_1 = [d[0] for d in contact_results]
        scores_2 = [d[1] for d in contact_results]
        scores_short = np.concatenate(scores_1)
        mscores_long = np.concatenate(scores_2)
        metrix_score = [scores_short, mscores_long]

    if metrix=='SS':
        # calculate dssp score
        ref_dssp_code = md.compute_dssp(ref,simplified=True)
        beta_lst = None
        prior_b_pair = None
        minor_b_pair = None
        if sum(np.isin(ref_dssp_code[0], 'E')):
            beta_index = np.where(ref_dssp_code[0]=='E')
            beta_lst = extract_consecutive_parts(beta_index[0])
            hbonds_lst = md.baker_hubbard(ref)
            res2atom = []
            for i in range(ref.top.n_residues):
                at1 = ref.top.select('resid '+str(i))[0]
                at2 = ref.top.select('resid '+str(i))[-1]
                res2atom.append([at1,at2])
            
            # Create a list of coarse-grained hydrogen bonds represented by pairs of amino acids.
            res_hbonds = []
            for hbond in hbonds_lst:
                donor = hbond[0]
                acc = hbond[2]
                res_hbond = [0,0]
                for n,at in enumerate(res2atom):
                    if at[0]<=donor<=at[1]:
                        res_hbond[0] = n
                    elif at[0]<=acc<=at[1]:
                        res_hbond[1] = n
                res_hbonds.append(np.array(res_hbond))
            res_hbonds = np.array(res_hbonds)
            res_hbonds_beta = []    # coarse-grained hbond represented by residue pairs
            for hbond in res_hbonds:
                if sum(np.isin(np.concatenate(beta_lst), hbond[0]))\
                    and sum(np.isin(np.concatenate(beta_lst), hbond[1])):
                    res_hbonds_beta.append(hbond)
            
            beta_pair = []
            for pair in res_hbonds_beta:
                ind = [-1, -1]
                for n,lst in enumerate(beta_lst):
                    if sum(np.isin(lst,pair[0])):
                        ind[0] = n
                    elif sum(np.isin(lst,pair[1])):
                        ind[1] = n
                beta_pair.append(np.sort(np.array(ind)))
            beta_pair = np.unique(beta_pair, axis=0)
            prior_b_pair = []
            minor_b_pair = []
            for p in beta_pair:
                start = beta_lst[p[0]][-1]+1
                end = beta_lst[p[1]][0]
                if sum(ref_dssp_code[0][start:end]=='H') or sum(ref_dssp_code[0][start:end]=='E'):
                    minor_b_pair.append(p)
                else:
                    prior_b_pair.append(p)

        pool = multiprocessing.Pool(processes=N)
        results = pool.map(cal_dssp_score, [(traj_slice,ref_dssp_code,beta_lst,prior_b_pair,minor_b_pair) for traj_slice in traj_slices])
        pool.close()
        pool.join()
        helix = [d[0] for d in results]
        sheet_1 = [d[1] for d in results]
        sheet_2 = [d[2] for d in results]
        helix_score = np.concatenate(helix)
        if sheet_1[0] is None:
            sheet_1_score = np.array([None])
        else:
            sheet_1_score = np.concatenate(sheet_1)
        if sheet_2[0] is None:
            sheet_2_score = np.array([None])
        else:
            sheet_2_score = np.concatenate(sheet_2)
        metrix_score = [helix_score, sheet_1_score, sheet_2_score]

    return metrix_score

# def make_slice_trajs(whole_traj):
#     # divide traj into N chunks
#     N = multiprocessing.cpu_count()
#     chunk_size = len(whole_traj) // N
#     chunk_remainder = len(whole_traj) % N
#     traj_slices = [whole_traj[i:i+chunk_size] for i in range(0, len(whole_traj)-chunk_remainder, chunk_size)]
#     if chunk_remainder > 0:
#         traj_slices.append(whole_traj[-chunk_remainder:])
#     return traj_slices

def make_slice_trajs(whole_traj):
    # divide traj into N chunks
    N = multiprocessing.cpu_count()
    chunk_size = len(whole_traj) // N
    chunk_remainder = len(whole_traj) % N
    traj_slices = [whole_traj[i:i+chunk_size] for i in range(0, len(whole_traj)-chunk_remainder, chunk_size)]
    if chunk_remainder>0 and len(traj_slices)==N:
        traj_slices[-1] = whole_traj[-(chunk_remainder+chunk_size):]
    elif chunk_remainder>0 and len(traj_slices)==N-1:
        traj_slices.append(whole_traj[-chunk_remainder:])
    return traj_slices

def make_slice(data):
    N = multiprocessing.cpu_count()
    if len(data)<N:
        return [[i] for i in data]
    else:
        chunk_size = len(data) // N
        chunk_remainder = len(data) % N
        slices = [data[i:i+chunk_size] for i in range(0, len(data)-chunk_remainder, chunk_size)]
        if chunk_remainder > 0:
            slices.append(data[-chunk_remainder:])
    return slices

def read_history_struc_score(whole_traj):

    if whole_traj is not None:
        metrixs=args.metrix.split('_')
        scorefiles = []
        for metrix in metrixs:
            if metrix=='SS':
                scorefiles.append('helix_ss.npy')
                scorefiles.append('sheet_1_ss.npy')
                scorefiles.append('sheet_2_ss.npy')
            elif metrix=='Q':
                scorefiles.append('q.npy')
            elif metrix=='R':
                scorefiles.append('rmsd.npy')
            elif metrix=='contact':
                scorefiles.append('short_contact.npy.npy')
                scorefiles.append('long_contact.npy')
            elif metrix=='AF':
                scorefiles.append('AF_score.npy')
            elif metrix=='U':
                scorefiles.append('U_score.npy')
        
        fileflag = True
        for f in scorefiles:
            fileflag &= os.path.isfile(f)
        if fileflag:
            scores = [np.load(f,allow_pickle=True) for f in scorefiles]
        else:
            traj_slices = make_slice_trajs(whole_traj)
            scores = []
            for metrix in metrixs:
                score = call_cal_struc_score(traj_slices, metrix=metrix)
                if metrix=='SS':
                    scores.append(score[0])
                    scores.append(score[1])
                    scores.append(score[2])
                elif metrix=='Q':
                    scores.append(score)
                elif metrix=='R':
                    scores.append(score)
                elif metrix=='AF':
                    scores.append(score)
                elif metrix=='U':
                    scores.append(score)
                elif metrix=='contact':
                    scores.append(score[0])
                    scores.append(score[1])
    else:
        scores = []

    return scores

def read_whole_traj(file_dirs_list, pdb, interval):
    """read a traj

    Args:
        file_dirs_list (_type_): _description_
        pdb (_type_): _description_

    Returns:
        _type_: _description_
    """
    trajs = []
    data_output = []
    for i in file_dirs_list:
        file_list = os.listdir(i)
        if(len(file_list) != 0):
            num_round = len(file_list)
            for n,file in enumerate(file_list):
                if file.endswith('.dcd'):
                    traj = md.load_dcd(i+'/'+file, top=pdb)
                elif file.endswith('.xyz'):
                    traj = md.load_xyz(i+'/'+file, top=pdb)
                elif file.endswith('.trr'):
                    traj = md.load_trr(i+'/'+file, top=pdb)
                elif file.endswith('.xtc'):
                    traj = md.load_xtc(i+'/'+file, top=pdb)
                elif file.endswith('.pdb'):
                    traj = md.load_pdb(i+'/'+file)
                res_frames = len(traj)%interval
                if res_frames==0:
                    continue
                else:
                    os.remove(i+'/'+file)
                    traj[:-res_frames].save_dcd(i+'/'+str(n)+'.dcd')
            whole_dirs = [i+'/'+str(j)+'.dcd' for j in range(num_round)]
            data_output = feature(whole_dirs, pdb, args.sampling_interval)
            for j in whole_dirs:
                trajs.append(md.load_dcd(j, top=pdb))
    with open('data_output.pkl', 'wb') as out_data:
        pickle.dump(data_output, out_data)
    if(len(trajs) > 1):
        whole_traj = md.join(trajs, check_topology=False)
    elif(len(trajs) == 1):
        whole_traj = trajs[0]
    else:
        whole_traj = None

    return whole_traj

def get_recent_files(folder_path, num_files):
    files = os.listdir(folder_path)
    files = [os.path.join(folder_path, file) for file in files]
    files = filter(os.path.isfile, files)
    files = sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)
    return files[:num_files]

def read_slice_traj(file_dir, pdb, nth_round):
    """construct a traj form sevral part trajs

    Args:
        file_dir (_type_): _description_
        pdb (_type_): _description_
        nth_round (_type_): _description_

    Raises:
        SystemExit: _description_

    Returns:
        _type_: _description_
    """
    interval = int(args.sampling_interval)
    trajs = []
    file_list = []
    for i in file_dir:
        ifile = get_recent_files(i, 1)
        # ifile=os.listdir(i)
        # print(ifile)
        # print(pdb)
        if len(ifile)==0:
            continue
        for n,file in enumerate(ifile):
            if file.endswith('.dcd'):
                traj = md.load_dcd(file, top=pdb)
            elif file.endswith('.xyz'):
                traj = md.load_xyz(file, top=pdb)
            elif file.endswith('.trr'):
                traj = md.load_trr(file, top=pdb)
            elif file.endswith('.xtc'):
                traj = md.load_xtc(file, top=pdb)
            elif file.endswith('.pdb'):
                traj = md.load_pdb(file)
            
            res_frames = len(traj)%interval
            if res_frames==0:
                continue
            else:
                os.remove(file)
                traj[:-res_frames].save_dcd(i+'/'+str(nth_round)+'.dcd')
        file_list.append(i+'/'+str(nth_round)+'.dcd')
    if(len(file_list)==0):
        whole_traj = None
        return whole_traj
    updated_data_output = feature(file_list, pdb, args.sampling_interval)

    if(os.path.isfile('data_output.pkl')):
        with open('data_output.pkl','rb') as in_data:
            data_output = pickle.load(in_data)
        for i in updated_data_output:
            data_output.append(i)
    else:
        raise SystemExit('The "data_output.pkl" file missing!')

    with open('data_output.pkl', 'wb') as out_data:
        pickle.dump(data_output, out_data)

    for i in file_list:
        trajs.append(md.load_dcd(i, top=pdb))

    whole_traj = md.join(trajs, check_topology=False)

    return whole_traj

def update_whole_traj(his_traj, nth_round):
    traj_dirs = args.traj_dirs
    dirs = os.listdir(traj_dirs)
    num_replica = len(dirs)
    whole_dirs = [traj_dirs+'/replica'+str(i) for i in range(num_replica)]
    traj = read_slice_traj(whole_dirs, args.pdb, nth_round)
    if traj is None:
        whole_traj = his_traj
    else:
        trajs = [his_traj, traj]
        whole_traj = md.join(trajs, args.pdb)
        traj.save_dcd('./updated_traj.dcd')

    return whole_traj, traj

def update_struc_score(history_scores, traj):
    traj_slices = make_slice_trajs(traj)
    metrixs=args.metrix.split('_')
    n = 0
    for metrix in metrixs:
        if metrix=='SS':
            n += 3
        elif metrix=='R':
            n += 1
        elif metrix=='Q':
            n += 1
        elif metrix=='AF':
            n += 1
        elif metrix=='U':
            n += 1
        elif metrix=='contact':
            n += 2
    scores = [np.array([])]*n
    
    n = 0
    for metrix in metrixs:
        score = call_cal_struc_score(traj_slices, metrix=metrix)
        #----------------------------------------------
        # dirty code
        if metrix=='SS':
            scores[n]=np.append(history_scores[n],score[0])
            scores[n+1]=np.append(history_scores[n+1],score[1])
            scores[n+2]=np.append(history_scores[n+2],score[2])
            n += 3
        elif metrix=='Q':
            scores[n]=np.append(history_scores[n],score)
            n += 1
        elif metrix=='R':
            scores[n]=np.append(history_scores[n],score)
            n += 1
        elif metrix=='AF':
            scores[n]=np.append(history_scores[n],score)
            n += 1
        elif metrix=='U':
            scores[n]=np.append(history_scores[n],score)
            n += 1
        elif metrix=='contact':
            scores[n]=np.append(history_scores[n],score[0])
            scores[n+1]=np.append(history_scores[n+1],score[1])
            n += 2
        #----------------------------------------------
    
    return scores

def DCD_append(base_file, append_file):
    updated_traj = parseDCD(append_file)
    w = DCDFile(base_file, mode='a')
    frame = w.next()
    unitcell = frame.getUnitcell()
    w.write(updated_traj, unitcell=unitcell)

def setup():
    if(os.path.isfile('history.dcd')):
        whole_traj = md.load_dcd('history.dcd', top=args.pdb)
        scores = read_history_struc_score(whole_traj)
        # update
        whole_traj, res_traj = update_whole_traj(whole_traj, nth_round=args.nth_round)
        if res_traj is not None:
            scores = update_struc_score(scores, res_traj)
    else:
        traj_dirs = args.traj_dirs
        dirs = os.listdir(traj_dirs)
        num_replica = len(dirs)
        whole_dirs = [traj_dirs+'/replica'+str(i) for i in range(num_replica)]
        pdb = args.pdb
        whole_traj = read_whole_traj(whole_dirs, pdb, args.sampling_interval)
        scores = read_history_struc_score(whole_traj)
    
    metrixs=args.metrix.split('_')
    scorefiles = []
    for metrix in metrixs:
        if metrix=='SS':
            scorefiles.append('helix_ss.npy')
            scorefiles.append('sheet_1_ss.npy')
            scorefiles.append('sheet_2_ss.npy')
        elif metrix=='Q':
            scorefiles.append('q.npy')
        elif metrix=='R':
            scorefiles.append('rmsd.npy')
        elif metrix=='contact':
            scorefiles.append('short_contact.npy.npy')
            scorefiles.append('long_contact.npy')
        elif metrix=='AF':
            scorefiles.append('AF_score.npy')
        elif metrix=='U':
            scorefiles.append('U_score.npy')

    if whole_traj is not None:
        if os.path.isfile('./updated_traj.dcd'):
            DCD_append(base_file='./history.dcd', append_file='./updated_traj.dcd')
        else:
            whole_traj.save_dcd('./history.dcd')

        for n,f in enumerate(scorefiles):
            np.save(f, scores[n])

        #shutil.rmtree(args.traj_dirs)
        #os.mkdir(args.traj_dirs)

    return whole_traj, scores

def select_states(scores,n_replicas,traj,native_ref,max_entropy=False,cutoff=1.0):
    if max_entropy:
        # check the similarity between the selected structures
        heavy = native_ref.top.select_atom_indices('heavy')
        check_rmsd = 0
        nth_score = 0
        flag_rmsd = True
        while flag_rmsd:
            tmp_score = np.sort(np.unique(scores))[nth_score]
            sel_index = np.where(scores<=tmp_score)[0]
            # put the minimum in first position
            min_index = np.where(scores[sel_index]==np.min(scores))[0][0]
            tmp_x = sel_index[0]
            sel_index[0] = sel_index[min_index]
            sel_index[min_index] = tmp_x

            sel_traj = traj[sel_index]
            # check_rmsd = md.rmsd(sel_traj,native_ref,frame=0,atom_indices=heavy)
            # min_id = np.where(check_rmsd==np.min(check_rmsd))[0]
            check_rmsd = md.rmsd(sel_traj,sel_traj[0],frame=0,atom_indices=heavy)
            if np.max(check_rmsd)>cutoff:
                sel_id = np.where(check_rmsd<=cutoff)[0]
                sel_index = sel_index[sel_id]
                flag_rmsd = False
            else:
                flag_rmsd = True
            nth_score += 1
            if nth_score >= np.min([np.unique(scores).shape[0], 3]):
                break
    else:
        sel_index = np.where(scores==np.min(scores))[0]
        
    # sel_index_0 = np.where(scores[0]==np.min(scores[0]))[0]

    if sel_index.shape[0] >= n_replicas:
        sel_traj = traj[sel_index]
    else:
        ind_state = scores.argsort()
        sel_index = ind_state[:n_replicas]
        sel_traj = traj[sel_index]
    sel_scores = scores[sel_index]
    return sel_index, sel_traj, sel_scores

def traj2sample_index(sel_traj,n_replicas):
    # ref_id = random.randint(0,len(sel_traj)-1)
    ref_id = 0
    reference = sel_traj[ref_id]
    if len(sel_traj)>n_replicas:
        sampled_states, sampled_indexs = sample_states(sel_traj,reference,n_replicas,n_iter=100)
    else:
        sampled_states = sel_traj
        sampled_indexs = list(range(len(sel_traj)))
    return sampled_states, sampled_indexs

def filter_index(score, percentage):
    cutoff = np.min(score) + percentage*(np.max(score)-np.min(score))
    n_sel = np.where(score<=cutoff)[0].shape[0]
    sel_score = np.sort(score)[:n_sel]
    sel_index = np.argsort(score)[:n_sel]
    return sel_score, sel_index                 

def filter_index_from_prior_score(score):
    metrixs=args.metrix.split('_')
    score_messages = []
    percentages = []
    for metrix in metrixs:
        if metrix=='SS':
            score_messages.append('Alpha-Helix')
            score_messages.append('Beta-Sheet-I')
            score_messages.append('Beta-Sheet-II')
            percentages += [0.5,0.5,0.5]
        elif metrix=='Q':
            score_messages.append('Native-Contacts')
            percentages += [0.5]
        elif metrix=='R':
            score_messages.append('Native-Rmsd')
            percentages += [0.5]
        elif metrix=='contact':
            score_messages.append('Short-range-Contacts')
            score_messages.append('Long-range-Contacts')
            percentages += [0.5,0.5]
        elif metrix=='AF':
            score_messages.append('AlphaFold-score')
            percentages += [0.5]
                
    score_cutoffs = []
    for i in score:
        if i.any() is not None:
            score_cutoffs.append(np.max(i)/10)
            
    upg_score = []
    upg_score_messages = []
    upg_percentages = []
    print('-------------------------------------------------------------------')
    print('Global Minimum Score')
    for n,i in enumerate(score):
        if i.any() is not None:
            upg_score.append(i)
            upg_score_messages.append(score_messages[n])
            upg_percentages.append(percentages[n])
            print('    ' + score_messages[n] + ' : ', np.min(i))
    print('------------------------------------')
    
    
    print('Selected Local')
    print('    Stage                number of states        minimum score')
    sel_score = upg_score[0]
    indices = np.arange(upg_score[0].shape[0])
    for n,i in enumerate(upg_score):
        if np.min(i)<score_cutoffs[n]:
            sel_score, sel_index = filter_index(i[indices], upg_percentages[n])
            indices = indices[sel_index]
            print('    ' + upg_score_messages[n] + '        ' + str(sel_index.shape[0]) + '              ', np.min(sel_score))
    print('-------------------------------------------------------------------')
    
    return sel_score, indices

def cal_weighted_scale(data, x):
    from scipy.stats import norm
    data = np.sort(data)
    a = np.min(data)
    b = np.max(data)
    uniform_data = np.arange(a,b,(b-a)/data.shape[0])
    indices = np.array([np.argmin(np.abs(data-i)) for i in uniform_data])

    # Calculate the PDF of the data distribution
    pdf = norm.pdf(data)

    # Normalize the PDF so that the maximum value is 1
    pdf /= np.max(pdf)
    pdf = pdf[indices]

    # Calculate the weights based on the PDF and the distance from c
    weights = 1 - pdf
    cumulative_weights = np.cumsum(weights)
    
    index = np.argmin(np.abs(uniform_data-x))
    weighted_scale = cumulative_weights[index]/cumulative_weights[-1]
    
    return weighted_scale

def left_skew(data):
    # Calculate the histogram
    hist, bins = np.histogram(data, bins='auto', density=True)
    mean = np.mean(data)
    left_data = data[data <= mean]
    left_indices = np.where(bins <= mean)[0]
    # left_skew = skew(left_data, bias=False)
    left_skewness = (mean - np.min(left_data))/np.mean(hist[left_indices]/np.max(hist))
    return left_skewness

def cal_loss(theta,cache_scores,number_replica,interval,k=1,method="left_skewness"):
    if method=="left_skewness":
        theta = np.array(theta).reshape(1,-1)
        new_kai = np.dot(theta, np.array(cache_scores[-number_replica*interval:])).reshape(-1,)
        kai = np.dot(theta, np.array(cache_scores)).reshape(-1,)
        x = np.min(new_kai)
        fx = cal_weighted_scale(kai, x)
        
        old_skewness = []
        new_skewness = []
        for n,score in enumerate(cache_scores):
            old_skewness.append(theta[0][n]*left_skew(score[:-number_replica*interval]))
            new_skewness.append(theta[0][n]*left_skew(score))
        old_skewness = np.array(old_skewness)
        new_skewness = np.array(new_skewness)      
            
        loss = fx - k*(np.sum(old_skewness)-np.sum(new_skewness))/np.sum(old_skewness)
        
        return loss
    elif method=="relative_entropy":
        default_resolution = 1000
        delt = [min_relative_std(x=score,interval=interval) for score in cache_scores]
        delt = np.array(delt).reshape(-1,1)
        d = np.max(delt)
        step = (1/d)/(default_resolution/2+2)
        alpha = np.array([np.concatenate([np.arange(0,step,step/(default_resolution/2+2))[1:-1],
                                         np.arange(0,1/d,step)[1:-1]]) for _ in range(len(cache_scores))])
        print('alpha_min=',np.min(alpha[0]),'alpha_max=',np.max(alpha[0]),'alpha.shape=',alpha.shape)
        theta = np.array(theta).reshape(-1,1)
        W = -np.log(alpha*delt) # W.shape = (N,default_resolution), N represents number of CVs
        W = W*theta
        W = W.T
        W = W/np.sum(W,axis=1).reshape(-1,1)
        R = np.dot(W,np.array(cache_scores)) # R.shape = (default_resolution,M), M represents number of samples
        print('R.shape=',R.shape)
        samples_index = np.argmin(R,axis=1)
        print('samples_index=',samples_index[:400])
        seed_arr = np.arange(0,cache_scores[0].shape[0],interval)
        sampled_alpha = []
        D_KL = []
        for n,i in enumerate(seed_arr): # this loop can be improved
            mask = np.where(samples_index>=i)[0]
            if mask.shape[0]==0:
                continue
            index = mask[np.argmin(samples_index[mask]-i)]
            if samples_index[index]-i<interval:
                sampled_alpha.append(alpha[0][index])
                d_kl = np.array([-directional_relative_entropy(old_data=score[-number_replica*interval:],
                                                     new_data=score[n*interval:(n+1)*interval])
                        for score in cache_scores]).reshape(1,-1)
                
                D_KL.append(np.dot(d_kl,theta))
        sampled_alpha = np.array(sampled_alpha)
        D_KL = np.array(D_KL).reshape(-1,)
        print("sampled_alpha:",sampled_alpha,"D_KL:",D_KL)
        
        return sampled_alpha, D_KL
    
def update_agent_factor(agent_factor_loss,method="relative_entropy"):
    if method=="left_skewness":
        x_bounds = (0.5,2)
        x = np.array(agent_factor_loss[0]).reshape(-1,)
        y = np.array(agent_factor_loss[1]).reshape(-1,)
        ts_gp = ThompsonSamplingGP(1, x_bounds=x_bounds, X=x, y=y, interval_resolution=1000)
        sampled_x, sampled_y = ts_gp.choose_next_samples()
        ALPHA = sampled_x[0]
    elif method=="relative_entropy":
        x_bounds = (agent_factor_loss[0][0],agent_factor_loss[0][-1])
        x = agent_factor_loss[0]
        y = agent_factor_loss[1]
        ts_gp = ThompsonSamplingGP(1, x_bounds=x_bounds, X=x, y=y, interval_resolution=1000)
        sampled_x, sampled_y = ts_gp.choose_next_samples()
        ALPHA = sampled_x[0]
        
    return ALPHA

def cal_latent(cluster_labels, score, history_actions):
    t = history_actions.shape[0]
    indices = np.array(history_actions).reshape(-1,)
    his_domain = cluster_labels[indices]
    count = Counter(his_domain)
    N_CG = np.zeros(cluster_labels.shape[0])
    for C in np.unique(his_domain):
        index = np.where(cluster_labels==C)[0]
        N_CG[index] += count[C]
    
    uncertainty = np.sqrt(N_CG/np.log(t))
    print('uncertainty (num_nonzero,min,mean,max):',np.where(uncertainty>0)[0].shape[0],np.min(uncertainty),np.mean(uncertainty),np.max(uncertainty))
    latent_score = score*np.exp(uncertainty)

    return latent_score

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="Calculate scoresfrom structure & MSM model",
                                epilog=""" """)

parser.add_argument('--metrix', dest='metrix', help='what metrix used for structrue scoring, supported method:"SS","Q","R","AF","contact", any combination, eg:"SS_AF"')
parser.add_argument('--advanced_methods', dest='advanced_methods', help='advanced sampling methods, supported method:"Multi_stage_TS","Adaptive_CV_TS","CVgenerator_TS","TS_GP"')
parser.add_argument('--prior', dest='prior', help='only need for "TS_GP" method')
parser.add_argument('--TS_GP_mode', dest='TS_GP_mode', help='only need for "TS_GP" method')
parser.add_argument('--alphafold_probs', dest='alphafold_probs', help='filename of alphafold_probability info')
parser.add_argument('--prior_rounds', dest='prior_rounds', type=int, help='number of rounds before advanced sampling, only used in Thompson Sampling')
parser.add_argument('--traj_dirs', dest='traj_dirs', help='the current traj dirs')
parser.add_argument('--pdb', dest='pdb', help='pdb structure file')
parser.add_argument('--nth_round', dest='nth_round', type=int, help='nth round')
parser.add_argument('--lagtime', dest='lagtime', type=int, help='lagtime for build MSM model')
parser.add_argument('--outpdb_dir', dest='outpdb_dir', help='output selected pdb struct as new starting point to run adaptive sampling')
parser.add_argument('--sampling_interval', dest='sampling_interval', type=int, help='sampling interval')
parser.add_argument('--n_clusters', dest='n_clusters', type=int, help='number of clusters')
parser.add_argument('--n_replicas', dest='n_replicas', type=int, help='number of replicas')
parser.add_argument('--verbose', default=False, action='store_true', dest='verbose', help='if specified, print verbose output')
args = parser.parse_args()
if args.traj_dirs is None:
    parser.error('No traj_dirs specified')
if args.pdb is None:
    parser.error('No pdb specified')
if args.lagtime is None:
    parser.error('No lagtime specified')
if args.outpdb_dir is None:
    parser.error('No outpdb_dir specified')
if args.sampling_interval is None:
    parser.error('No sampling_interval specified')

start = datetime.now()
## read current structure scores
num_cores = multiprocessing.cpu_count()
whole_traj, scores = setup()
if whole_traj is None:
    raise SystemExit('empty trajectority!')

if(os.path.isfile('con.pkl')):
    with open('con.pkl','rb') as in_data:
        connections = pickle.load(in_data)
else:
    connections = []
if(os.path.isfile('con_frame.pkl')):
    with open('con_frame.pkl','rb') as in_data:
        connections_with_frame = pickle.load(in_data)
else:
    connections_with_frame = []
    
if(os.path.isfile('his_actions.pkl')):
    with open('his_actions.pkl','rb') as in_data:
        his_actions = pickle.load(in_data)
else:
    his_actions = [] # frame index in whole_traj as history information used to bias the sampling process
    
interval = args.sampling_interval


ad_methods = ['Multi_stage_TS','Adaptive_CV_TS','CVgenerator_TS'] # CVgenerator_TS is what im using now !!!!!
if args.advanced_methods in ad_methods:
    if args.advanced_methods == 'Multi_stage_TS':
        # Multi-stages sampling
        sel_score, sel_index = filter_index_from_prior_score(scores)
    elif args.advanced_methods == 'Adaptive_CV_TS':
        # Adaptive CV TS
        metrixs=args.metrix.split('_')
        M = []
        for i in metrixs:
            if i == 'SS':
                M += ['SS-alpha','SS-helix1','SS-helix2']
            elif i == 'contact':
                M += ['Contact-short','Contact-long']
            else:
                M += [i]
        cache_scores = []
        matrixs_name = []
        for n,i in enumerate(scores):
            if i.any() is None:
                continue
            else:
                cache_scores.append(i)
                matrixs_name.append(M[n])
        delt = []
        for score in cache_scores:
            if score.shape[0]<interval*args.n_replicas:
                delt = [random.randint(1, 100) for _ in range(len(cache_scores))]
                break
            # # diff
            # delt.append(np.abs(np.min(score)-np.min(score[:-interval*num_cores])))
            # relative fluctuation
            delt.append(cal_relative_fluctuation(score))
        if np.max(delt)==0:
            delt = [random.randint(1, 100) for _ in range(len(cache_scores))]
        delt = np.array(delt).reshape(-1,)
        CV_id = np.argmax(delt)
        print('*********** Select ',matrixs_name[CV_id],'as metrix ***********')
        sel_score = cache_scores[CV_id]
        sel_index = np.arange(sel_score.shape[0])
    elif args.advanced_methods == 'CVgenerator_TS': # this is what we are using now !!!!!
        from scipy.stats import skew
        native_ref = md.load_pdb(args.pdb)
        
        # parameter setting
        windows_size = [5,8,12,16,20,24,28,32]
        steps = [3,4,6,8,10,12,14,16]
        
        cache_scores = []
        cvgenerator_ts = CVgenerator(args.pdb,windows_size='auto',steps='auto')
        sel_indices, CV_residues_index = cvgenerator_ts.generateCV()
        prior_theta = compute_prior_theta(CV_residues_index)
        prior_theta = prior_theta/np.sum(prior_theta)
        print('whole_traj length=',len(whole_traj))
        
        if args.nth_round==0:
            sel_trajs = whole_traj
        else:
            sel_trajs = whole_traj[-interval*args.n_replicas:]
        traj_slices = make_slice_trajs(sel_trajs)
        print('len sel_trajs=',len(sel_trajs))
        print('len_traj_slices=',len(traj_slices))
        print("traj_slices=",traj_slices)
        
        # # ----------------------------------------------------------------------
        # #             rmsd metrix for each CV_descriptor
        # cv_gen_rmsd = []
        # for sel_indice in sel_indices:
        #     pool = multiprocessing.Pool(processes=args.n_replicas)
        #     rmsd_results = pool.map(cal_rmsd_score, [(traj_slice, native_ref, sel_indice) for traj_slice in traj_slices])
        #     pool.close()
        #     pool.join()
        #     rmsd_value = np.concatenate(rmsd_results)
        #     cv_gen_rmsd.append(rmsd_value)
        # if args.nth_round==0:
        #     cache_scores += cv_gen_rmsd
        #     with open('CV_gen_rmsd.pkl','wb') as w:
        #         pickle.dump(cv_gen_rmsd,w)
        # else:
        #     with open('CV_gen_rmsd.pkl','rb') as r:
        #         CV_gen_rmsd = pickle.load(r)
        #     for i in range(len(CV_gen_rmsd)):
        #         CV_gen_rmsd[i] = np.append(CV_gen_rmsd[i],cv_gen_rmsd[i])
        #     cache_scores += CV_gen_rmsd
        #     with open('CV_gen_rmsd.pkl','wb') as w:
        #         pickle.dump(CV_gen_rmsd,w)
        
        # ----------------------------------------------------------------------
        #             max-adjacency-bonds metrix for each CV_descriptor
        cv_gen_mab = []
        
        indices_slices = make_slice_trajs(sel_indices)
        pool = multiprocessing.Pool(processes=args.n_replicas)
        mab_results = pool.map(cal_mab_score, [(sel_trajs, args.pdb, indices_slice) for indices_slice in indices_slices])
        pool.close()
        pool.join()
        mab_value = np.concatenate(mab_results,axis=0)
        cv_gen_mab = list(mab_value)
        
        # for sel_indice in sel_indices:
        #     pool = multiprocessing.Pool(processes=args.n_replicas)
        #     mab_results = pool.map(cal_mab_score, [(traj_slice, args.pdb, max_num_bonds, cutoff, sel_indice) for traj_slice in traj_slices])
        #     pool.close()
        #     pool.join()
        #     mab_value = np.concatenate(mab_results)
        #     cv_gen_mab.append(mab_value)
        if args.nth_round==0:
            cache_scores = cv_gen_mab
            with open('CV_gen_mab.pkl','wb') as w:
                pickle.dump(cv_gen_mab,w)
        else:
            with open('CV_gen_mab.pkl','rb') as r:
                CV_gen_mab = pickle.load(r)
            for i in range(len(CV_gen_mab)):
                CV_gen_mab[i] = np.append(CV_gen_mab[i],cv_gen_mab[i])
            cache_scores = CV_gen_mab
            with open('CV_gen_mab.pkl','wb') as w:
                pickle.dump(CV_gen_mab,w)
        
        
        CVgen = True
        
        agent = 'learn_alpha'
        reduce_uncertainty = True
        # agent = 'optimize_alpha'
        # agent = 'learn_W'
        # agent = 'predict'
        if CVgen:
            # CV_gen
            theta = []
            delt = []
            for n,score in enumerate(cache_scores):
                if np.min(score)==0:
                    theta.append(-np.log(0.000001))
                else:
                    theta.append(-np.log(np.min(score)))
                
                arg_x = np.argmin(score)
                nth_traj = arg_x//interval
                seg_data = score[nth_traj*interval:(nth_traj+1)*interval]
                left_skewness = left_skew(seg_data)
                # if left_skewness<0.5:
                #     delt.append(0)
                # else:
                delt.append(left_skewness)
            delt = np.array(delt).reshape(-1,)
            theta = theta-(1.1*np.min(theta))
            theta = prior_theta*np.array(theta)
            theta = theta/np.sum(theta)
            print('theta=',theta)
            
            if agent == 'learn_alpha':
                # if(os.path.isfile('agent_factor_loss.pkl')):
                #     with open('agent_factor_loss.pkl', 'rb') as r:
                #         agent_factor_loss = pickle.load(r)
                #     if len(agent_factor_loss[0])==len(agent_factor_loss[1])+1:
                #         loss = cal_loss(theta,cache_scores,args.n_replicas,interval)
                #         agent_factor_loss[1].append(loss)
                #     if np.sum(delt)==0:
                #         ALPHA = 1.0
                #     else:
                #         ALPHA = update_agent_factor(agent_factor_loss)
                #         agent_factor_loss[0].append(ALPHA)
                # else:
                #     agent_factor_loss = [[],[]]
                #     ALPHA = 1.0
                #     agent_factor_loss[0].append(ALPHA)
                # with open('agent_factor_loss.pkl', 'wb') as w:
                #     pickle.dump(agent_factor_loss,w)
                ALPHA = 3/np.abs(np.dot(theta.reshape(1,-1),delt.reshape(-1,1)))
                print('Optimized ALPHA =',ALPHA)
                
                # ALPHA = 0.3
                W = []
                for n,i in enumerate(theta):
                    # # exponential factor
                    # W.append(i**(1-ALPHA*delt[n]))
                    # linear factor
                    W.append(i*np.exp(ALPHA*delt[n]))
                W = np.array(W)/np.sum(W)
                W = W.reshape(1,-1)
                print('W=',W)
                with open('weighted_factor.npy', 'ab') as w:
                    np.save(w,W.reshape(-1,))
                CVgen_score_pre = np.dot(W, np.array(cache_scores)).reshape(-1,)
                CVgen_score = CVgen_score_pre
                CV_indices = np.arange(cache_scores[0].shape[0])                    
                
                if reduce_uncertainty and len(his_actions)>0:
                    # calculate latent score for all CV_scores
                    actions_in_reduced_region = np.intersect1d(CV_indices,np.concatenate(his_actions))
                    if actions_in_reduced_region.shape[0]>0:
                        n_clusters = args.n_clusters
                        if(os.path.isfile('parameters.pkl')):
                            with open('parameters.pkl','rb') as r:
                                n_clusters = pickle.load(r)
                            n_clusters += 10
                        mapped_actions = np.array([np.where(CV_indices==action)[0] for action in actions_in_reduced_region]).reshape(-1,)
                        sel_traj = whole_traj[CV_indices]
                        sel_CV_score = CVgen_score_pre[CV_indices]
                        cluster_labels = cluster(sel_traj.xyz,n_clusters)
                        latent_scores = cal_latent(cluster_labels,sel_CV_score,mapped_actions)
                        CVgen_score = latent_scores
                        print('global cluster number is ',n_clusters)
                        with open('parameters.pkl','wb') as w:
                            pickle.dump(n_clusters,w)
                # CVgen_score = sel_CV_score 
                # filter CV_score
                memory_cutoff = 50000
                if CV_indices.shape[0]>memory_cutoff:
                    CV_indices = np.argsort(CVgen_score)[:memory_cutoff]
                    CVgen_score = CVgen_score[CV_indices]
                print('CVgen_score (min,mean,max):',np.min(CVgen_score_pre),np.mean(CVgen_score_pre),np.max(CVgen_score_pre))              
            elif agent == 'predict':
                theta = theta.reshape(1,-1)
                base_score = np.dot(theta, np.array(cache_scores))
                instant_optimal_arg = np.argmin(base_score)
                dis = np.array(cache_scores) - np.array(cache_scores)[:,instant_optimal_arg].reshape(-1,1)
                predict_score = np.dot(theta, dis)
                CVgen_score = predict_score
                CV_indices = np.arange(CVgen_score.shape[0])
            else:
                theta = theta.reshape(1,-1)
                CVgen_score = np.dot(theta, np.array(cache_scores))
                CV_indices = np.arange(CVgen_score.shape[0])
            sel_score = CVgen_score.reshape(-1,)
            sel_index = CV_indices
            arms_values, arms_indices = thompson_sampling_MSM(args.lagtime,200,prior_score=sel_score,prior_score_indices=sel_index)
            samples_index = choose_action(arms_values, arms_indices, 8, args.n_replicas, max_score=False)
        else:
            # useful method but waste time
            delt = []
            for score in cache_scores:
                if score.shape[0]<interval*args.n_replicas:
                    delt.append(cal_relative_fluctuation(score))
                elif score.shape[0]<2*interval*args.n_replicas:
                    delt.append(np.abs(np.min(score)-np.min(score[:-interval*args.n_replicas]))/np.mean(score[:-interval*args.n_replicas]))
                else:
                    delt.append(np.abs(np.min(score)-np.min(score[:-interval*args.n_replicas]))/np.mean(score[-2*interval*args.n_replicas:-interval*args.n_replicas]))
                # # relative fluctuation
                # delt.append(cal_relative_fluctuation(score))
            if np.max(delt)==0:
                print('all delt is 0, maybe in local minimum!')
                delt = [random.randint(1, 100) for _ in range(len(cache_scores))]
                
            delt = np.array(delt).reshape(-1,)
            CV_id = np.argmax(delt)
            print('*********** Select '+str(CV_id)+'-th CV as metrix ***********')
            if cache_scores[0].shape[0]<interval*args.n_replicas:
                sel_score = cache_scores[CV_id]
                sel_index = np.arange(sel_score.shape[0])
            else:
                sel_score = cache_scores[CV_id][-interval*args.n_replicas:]
                sel_index = np.arange(sel_score.shape[0])+cache_scores[CV_id].shape[0]-interval*args.n_replicas
            arms_values, arms_indices = thompson_sampling_MSM(args.lagtime,args.n_clusters,prior_score=sel_score,prior_score_indices=sel_index)
            samples_index = choose_action(arms_values, arms_indices, 8, args.n_replicas, max_score=False)
        
        
        ## design a multi direction sampling stratagy
        # CV_weights = delt/np.sum(delt)
        # samples_assign = allocate_elements(CV_weights,num_cores)
        # print('CV_index : number samples')
        # for i in range(samples_assign.shape[0]):
        #     print(i,' : ',samples_assign[i])
        # samples_index = []
        # for CV_id in range(samples_assign.shape[0]):
        #     if cache_scores[0].shape[0]<interval*num_cores:
        #         sel_score = cache_scores[CV_id]
        #     else:
        #         sel_score = cache_scores[CV_id][-interval*num_cores:]
        #     sel_index = np.arange(sel_score.shape[0])
        #     arms_values, arms_indices = thompson_sampling_MSM(args.lagtime,args.n_clusters,prior_score=sel_score,prior_score_indices=sel_index)
        #     samples_index += choose_action(arms_values, arms_indices, 8, samples_assign[CV_id], max_score=False)
            
        
    
    if args.prior_rounds <= args.nth_round:
        native_ref = md.load_pdb(args.pdb)
        heavy = native_ref.top.select_atom_indices('heavy')
        alpha = native_ref.top.select_atom_indices('alpha')
        all = native_ref.top.select_atom_indices('all')
        # BIO_FEAT = []
        # trajs = [whole_traj[i*interval:(i+1)*interval] for i in range(int(len(whole_traj)/interval))]
        # for traj in trajs:
        #     bio_feat = []
        #     rmsd = md.rmsd(traj, reference=native_ref, atom_indices=heavy)
        #     bio_feat.append(rmsd)
        #     q = best_hummer_q(traj, native_ref)
        #     bio_feat.append(q)
        #     bio_feat = np.array(bio_feat).T
        #     BIO_FEAT.append(bio_feat)
        # BIO_FEAT = np.array(BIO_FEAT)
        
        # arms_values, dtrajs = thompson_sampling_MSM(args.lagtime,args.n_clusters,bio_feat=BIO_FEAT)
        # samples_index = choose_bandit(arms_values, dtrajs, 1, 4, num_cores)
        
        # if args.advanced_methods == 'CVgenerator_TS':
        #     # CV_generator reward function
        #     CV_weights = []
        #     deviations = []
        #     for score in cache_scores:
        #         CV_weights.append(cal_relative_fluctuation(score))
        #         deviations.append(1-score/np.mean(score))
        #     CV_weights = np.array(CV_weights)
        #     CV_weights = CV_weights/np.sum(CV_weights)
        #     deviations = np.array(deviations)
        #     tica = get_tica(1,args.lagtime)
        #     rewards = CV_gen_func(CV_weights,deviations,tica)
        #     rewards_con = np.concatenate(rewards)
        #     all_index = np.argsort(rewards_con)%rewards.shape[1]
        #     nn = num_cores
        #     samples_index = []
        #     while len(samples_index)<num_cores:
        #         samples_index = list(np.unique(all_index[-nn:]))
        #         nn += 1
        # else:
        #     arms_values, arms_indices = thompson_sampling_MSM(args.lagtime,args.n_clusters,prior_score=sel_score,prior_score_indices=sel_index)
        #     samples_index = choose_action(arms_values, arms_indices, 8, 32, max_score=False)
        
        
        his_actions.append(samples_index)
        
        # rmsd = md.rmsd(whole_traj, native_ref, frame=0, atom_indices=all)
        # whole_traj[np.argmin(rmsd)].save_pdb('min_rmsd.pdb')
        # print("Minum rmsd is",np.min(rmsd))
        
        mab = cal_mab_score((whole_traj, args.pdb, [sel_indices[-1]]))
        whole_traj[np.argmin(mab)].save_pdb('min_mab.pdb')
        print("Minimum mab-score is",np.min(mab))
        
        os.mkdir(args.outpdb_dir + '/round' + str(args.nth_round + 1) + '/')
        sampled_states = whole_traj[samples_index]
        for i, state in enumerate(sampled_states):
            state.save_pdb(args.outpdb_dir + '/round' + str(args.nth_round + 1) + '/' + str(i) + '.pdb')
            node = samples_index[i] // interval
            frame = samples_index[i] % interval
            next_node = (len(whole_traj) // interval) + i
            connections.append((node, next_node))
            connections_with_frame.append([node, frame, next_node])
    else:
        struc_falg = False
        if np.min(scores[-1])<0.01:
            struc_falg = True

        if struc_falg:
            print('sampling low count states from MSM')
            transition_score, dtrajs = low_count_MSM(args.lagtime,args.n_clusters)
            #print('length:', len(set(transition_score)), 'transition score:', set(transition_score))
                
            ind_state = transition_score.argsort()
            selected_state = [ind_state[0]]
            num_selected = 1
            n = 1
            cluster_set = set()
            cluster_set.add(dtrajs[ind_state[0]])
            while num_selected < args.n_replicas:
                ind = ind_state[n]
                if dtrajs[ind] in cluster_set:
                    n += 1
                    continue
                else:
                    num_selected += 1
                    n += 1
                    cluster_set.add(dtrajs[ind])
                    selected_state.append(ind)
                    #print('score_set is :', score_set)


            # shutil.rmtree(args.outpdb_dir)
            os.mkdir(args.outpdb_dir + '/round' + str(args.nth_round + 1) + '/')
            for i, j in enumerate(selected_state):
                whole_traj[j].save_pdb(args.outpdb_dir + '/round' + str(args.nth_round + 1) + '/' + str(i) + '.pdb')
                node = j // interval
                frame = j % interval
                next_node = (len(whole_traj) // interval) + i
                connections.append((node, next_node))
                connections_with_frame.append([node, frame, next_node])

        else:
            native_ref = md.load_pdb(args.pdb)
            metrixs=args.metrix.split('_')
            score_messages = []
            for metrix in metrixs:
                if metrix=='SS':
                    score_messages.append('Alpha-Helix')
                    score_messages.append('Beta-Sheet-I')
                    score_messages.append('Beta-Sheet-II')
                elif metrix=='Q':
                    score_messages.append('Native-Contacts')
                elif metrix=='R':
                    score_messages.append('Native-Rmsd')
                elif metrix=='contact':
                    score_messages.append('Short-range-Contacts')
                    score_messages.append('Long-range-Contacts')
                elif metrix=='AF':
                    score_messages.append('AlphaFold-score')
                        
            score_cutoffs = []
            for i in scores:
                if i.any() is not None:
                    score_cutoffs.append(np.max(i)/10)
                    
            upg_score = []
            upg_score_messages = []
            print('-------------------------------------------------------------------')
            print('Global Minimum Score')
            for n,i in enumerate(scores):
                if i.any() is not None:
                    upg_score.append(i)
                    upg_score_messages.append(score_messages[n])
                    print('    ' + score_messages[n] + ' : ', np.min(i))
            print('------------------------------------')
            
            
            print('Selected Local')
            print('    Stage                number of states        minimum score')
            sel_score = upg_score[0]
            indices = np.arange(upg_score[0].shape[0])
            for n,i in enumerate(upg_score):
                if np.min(i)<score_cutoffs[n]:
                    sel_index, sel_traj, sel_score = select_states(i[indices],args.n_replicas,whole_traj[indices],native_ref,max_entropy=False)
                    sampled_states, sampled_indexs = traj2sample_index(sel_traj,args.n_replicas)
                    sampled_indexs_in_whole_traj = sel_index[sampled_indexs]
                    indices = indices[sel_index]
                    print('    ' + upg_score_messages[n] + '        ' + str(sel_index.shape[0]) + '              ', np.min(sel_score))
            print('-------------------------------------------------------------------')
                            
                            
            # ---------------------------------------------------------------------------------
            # this part for checking the "sampled_indexs_in_whole_traj", can be commented out if correct
            for n in range(len(sampled_states)):
                err = np.sum(sampled_states[n].xyz-whole_traj[sampled_indexs_in_whole_traj[n]].xyz)
                if err!=0:
                    print('error is', err)
                    raise SystemExit('sampled states index does not match the "whole_traj"!')
            # ---------------------------------------------------------------------------------

            his_actions.append(sampled_indexs_in_whole_traj)
            os.mkdir(args.outpdb_dir + '/round' + str(args.nth_round + 1) + '/')
            for i, state in enumerate(sampled_states):
                state.save_pdb(args.outpdb_dir + '/round' + str(args.nth_round + 1) + '/' + str(i) + '.pdb')
                node = sampled_indexs_in_whole_traj[i] // interval
                frame = sampled_indexs_in_whole_traj[i] % interval
                next_node = (len(whole_traj) // interval) + i
                connections.append((node, next_node))
                connections_with_frame.append([node, frame, next_node])

with open('con.pkl','wb') as out_data:
    pickle.dump(connections, out_data)
with open('con_frame.pkl','wb') as out_data_:
    pickle.dump(connections_with_frame, out_data_)
with open('his_actions.pkl','wb') as out_data:
    pickle.dump(his_actions, out_data)
# np.save('con.npy', connections)
# np.save('con_frame.npy', connections_with_frame)


end = datetime.now()
elapsed = end-start
time = elapsed.seconds + elapsed.microseconds*1e-6
print('Time consuming in computing SCORE = ', time)

del whole_traj