"""_summary_
    modified based on scheme.py
    
    2 changed from shceme.py:
        1. 2 stages for dssp
        2. scoring rule:1-(award-punish)
        
"""

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
    def __init__(self, pdb_structure, windows_size='auto', steps='auto', mode='regular'):
        self.struc = pdb_structure
        struc = md.load_pdb(pdb_structure)
        n_residues = struc.topology.n_residues
        if windows_size=='auto' and steps=='auto':
            self.windows_size = np.arange(5,n_residues,4)
            self.steps = np.round(self.windows_size/2).astype(int)
            # self.steps = np.array([2]*self.windows_size.shape[0])
            if self.windows_size[-1]!=n_residues:
                self.windows_size = np.append(self.windows_size,n_residues)
                self.steps = np.append(self.steps,n_residues)
            
        else:
            self.windows_size = windows_size
            self.steps = steps

        self.mode = mode
        
    def generateCV(self):
        struc = md.load_pdb(self.struc)
        n_residues = struc.topology.n_residues
        ref_dssp_code = md.compute_dssp(struc, simplified=False)
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
        heavy = struc.topology.select_atom_indices('heavy')
        alpha = struc.topology.select_atom_indices('alpha')
        important_atoms = heavy
        CV_indices = []
        CV_residues_index = []
        CV_struc_indices = []
        exclude_beta_strand = False
        if self.mode=='regular':
            if exclude_beta_strand:
                arr_scope = np.where(ref_dssp_code[0]!='E')[0]
                arrs_in = extract_consecutive_parts(arr_scope)
                filt_arrs_in = [i for i in arrs_in if i.shape[0]>3]
                for size,step in zip(self.windows_size,self.steps):
                    arrs_out = []
                    for arr in filt_arrs_in:
                        arrs_out += self.generate_sliding_windows(arr,size,step)
                    for ress in arrs_out:
                        resi_indices = struc.topology.select('resi '+' '.join(str(i) for i in ress))
                        if not any(np.array_equal(resi_indices, arr) for arr in CV_struc_indices):
                            CV_struc_indices.append(resi_indices)
                        sel_indices = np.intersect1d(important_atoms,resi_indices)
                        if not any(np.array_equal(sel_indices, arr) for arr in CV_indices):
                            CV_indices.append(sel_indices)
                            CV_residues_index.append(ress)
            else:
                arrs_in = np.arange(n_residues)
                for size,step in zip(self.windows_size,self.steps):
                    arrs_out = self.generate_sliding_windows(arrs_in,size,step)
                    for ress in arrs_out:
                        resi_indices = struc.topology.select('resi '+' '.join(str(i) for i in ress))
                        if not any(np.array_equal(resi_indices, arr) for arr in CV_struc_indices):
                            CV_struc_indices.append(resi_indices)
                        sel_indices = np.intersect1d(important_atoms,resi_indices)
                        if not any(np.array_equal(sel_indices, arr) for arr in CV_indices):
                            CV_indices.append(sel_indices)
                            CV_residues_index.append(ress)
            if flag_beta:
                cv_indices, cv_struc_indices, cv_residues_index  = self.find_beta(struc, beta_res, important_atoms)
                CV_indices += cv_indices
                CV_residues_index += cv_residues_index
                CV_struc_indices += cv_struc_indices
            
        elif self.mode=='structure':            
            if flag_alpha:
                cv_indices, cv_struc_indices, cv_residues_index = self.find_alpha(struc, alpha_res, important_atoms)
                CV_indices += cv_indices
                CV_residues_index += cv_residues_index
                CV_struc_indices += cv_struc_indices
            if flag_beta:
                cv_indices, cv_struc_indices, cv_residues_index  = self.find_beta(struc, beta_res, important_atoms)
                CV_indices += cv_indices
                CV_residues_index += cv_residues_index
                CV_struc_indices += cv_struc_indices
        if not os.path.isdir('CV_structures'):
            os.mkdir('CV_structures')
            for n,indice in enumerate(CV_struc_indices):
                CV_struc = struc.atom_slice(indice)
                CV_struc.save_pdb('./CV_structures/CV_structure_'+str(n)+'.pdb')    

        return CV_indices, CV_residues_index
        
    def generate_sliding_windows(self, arr_in, window_size, step):
        if len(arr_in) <= window_size:
            arrs_out = [list(arr_in)]
        else:
            arrs_out = []
            for i in range(0, len(arr_in) - window_size + 1, step):
                window = arr_in[i:i+window_size]
                arrs_out.append(list(window))
            
            # Include the last window if it is not complete
            # if len(arr_in) % step != 0:
            #     window = arr_in[-window_size:]
            #     arrs_out.append(list(window))
            
            if arrs_out[-1][-1] != arr_in[-1]:
                window = arr_in[-window_size:]
                arrs_out.append(list(window))
        
        return arrs_out
    
    # def generate_sliding_windows(self, arr_in, window_size, step):
    #     if len(arr_in) <= window_size:
    #         lst_out = [list(arr_in)]
    #     else:
    #         lst_out = []
    #         arr_length = len(arr_in)
    #         start = 0
    #         while start < arr_length:
    #             end = start + window_size
    #             # Wrap around to the beginning if the end exceeds the array length
    #             if end > arr_length:
    #                 window = np.concatenate((arr_in[start:], arr_in[:end - arr_length]))
    #             else:
    #                 window = np.array(arr_in[start:end])
    #             lst_out.append(window.tolist())
    #             start += step
            
    #     return lst_out
    
    def find_alpha(self, struc, alpha_res, important_atoms):
        alpha_level_struc = []
        CV_indices = []
        CV_residues_index = []
        CV_struc_indices = []
        for size,step in zip(self.windows_size,self.steps):
            for ress in alpha_res:
                if size>=ress.shape[0]:
                    if any(np.array_equal(ress, np.array(arr)) for arr in alpha_level_struc):
                        continue
                    else:
                        alpha_level_struc.append(list(ress))
                else:
                    arrs_out = self.generate_sliding_windows(ress,size,step)
                    alpha_level_struc += arrs_out
        for single_alpha in alpha_res:
            if not any(np.array_equal(single_alpha, np.array(arr)) for arr in alpha_level_struc):
                alpha_level_struc.append(single_alpha)

        n_seg = len(alpha_res)
        if n_seg>=2:
            for seg in range(2,n_seg+1):
                for i in range(0,n_seg-seg+1):
                    window = np.arange(alpha_res[i][0],alpha_res[i+seg-1][-1]+1)
                    alpha_level_struc.append(list(window))
        alpha_indices = []
        for ress in alpha_level_struc:
            resi_indices = struc.topology.select('resi '+' '.join(str(i) for i in ress))
            CV_struc_indices.append(resi_indices)
            sel_indices = np.intersect1d(important_atoms,resi_indices)
            alpha_indices.append(sel_indices)
            CV_residues_index.append(ress)
        CV_indices += alpha_indices
        return CV_indices, CV_struc_indices, CV_residues_index
    
    def find_beta(self, struc, beta_res, important_atoms):
        hbonds_lst = md.baker_hubbard(struc,distance_cutoff=0.25)
        res2atom = []
        CV_indices = []
        CV_residues_index = []
        CV_struc_indices = []
        for i in range(struc.top.n_residues):
            at1 = struc.top.select('resid '+str(i))[0]
            at2 = struc.top.select('resid '+str(i))[-1]
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
            if sum(np.isin(np.concatenate(beta_res), hbond[0]))\
                and sum(np.isin(np.concatenate(beta_res), hbond[1])):
                res_hbonds_beta.append(hbond)
        beta_pair = []
        for pair in res_hbonds_beta:
            ind = [-1, -1]
            for n,lst in enumerate(beta_res):
                if sum(np.isin(lst,pair[0])):
                    ind[0] = n
                elif sum(np.isin(lst,pair[1])):
                    ind[1] = n
            beta_pair.append(np.sort(np.array(ind)))
        beta_pair = np.unique(beta_pair, axis=0)

        beta_strand_lens = [i.shape[0] for i in beta_res]
        max_beta_strand = np.max(beta_strand_lens)
        beta_windows_sizes = np.arange(4,max_beta_strand+2,2)
        beta_steps = np.round(beta_windows_sizes/2).astype(int)
        beta_level_struc = []
        for pair in beta_pair:
            # align the beta-strand, make them same length
            sel_segs = [beta_res[pair[0]],beta_res[pair[1]]]
            iden = [beta_res[pair[0]].shape[0],beta_res[pair[1]].shape[0]]
            if beta_res[pair[0]].shape[0] != beta_res[pair[1]].shape[0]:
                big_seg = sel_segs[np.argmax(iden)]
                small_seg = sel_segs[np.argmin(iden)]
                
                seg_len = small_seg.shape[0]
                segs = self.generate_sliding_windows(big_seg,seg_len,1)
                segs_dis = []
                for seg in segs:
                    import itertools
                    pairs = list(itertools.product(seg, small_seg))
                    distances = md.compute_contacts(struc, pairs)
                    segs_dis.append(np.mean(np.min(distances[0].reshape(seg_len,seg_len),axis=1)))
                sel_seg = segs[np.argmin(segs_dis)]
                sel_segs = [small_seg,sel_seg]
            # align the beta-strand, make them paralelle aligned
            pair_res_dis1 = md.compute_contacts(struc, [(sel_segs[0][0],sel_segs[1][0])])
            pair_res_dis2 = md.compute_contacts(struc, [(sel_segs[0][0],sel_segs[1][-1])])
            if pair_res_dis2 < pair_res_dis1:
                sel_segs[1] = sel_segs[1][::-1]
                
            
            for size,step in zip(beta_windows_sizes,beta_steps):
                if size>=sel_segs[0].shape[0]:
                    if any(np.array_equal(np.concatenate(sel_segs), np.array(arr)) for arr in beta_level_struc):
                            continue
                    else:
                        beta_level_struc.append(list(np.concatenate(sel_segs)))
                else:
                    arrs_out1 = self.generate_sliding_windows(sel_segs[0],size,step)
                    arrs_out2 = self.generate_sliding_windows(sel_segs[1],size,step)
                    beta_level_struc += [list(np.concatenate([arrs_out1[i],arrs_out2[i]])) for i in range(len(arrs_out1))]

        beta_indices = []
        for ress in beta_level_struc:
            resi_indices = struc.topology.select('resi '+' '.join(str(i) for i in ress))
            CV_struc_indices.append(resi_indices)
            sel_indices = np.intersect1d(important_atoms,resi_indices)
            beta_indices.append(sel_indices)
            CV_residues_index.append(ress)
        CV_indices += beta_indices
        return CV_indices, CV_struc_indices, CV_residues_index

# CV_generator reward function
def CV_gen_func(weights, deviation, tica):
    import copy
    rewards = (weights*(deviation.T)).T
    hist, bins = np.histogram(tica, bins=500, density=True)
    hist[np.where(hist==0)] = 1/99999999999
    pseudo_free_energy = copy.deepcopy(tica)
    for i in range(bins.shape[0]-1):
        ind1 = np.where(pseudo_free_energy>=bins[i])
        ind2 = np.where(pseudo_free_energy<bins[i+1])
        ind = np.intersect1d(ind1, ind2)
        pseudo_free_energy[ind] = -np.log(hist[i])
    pseudo_free_energy[-1] = -np.log(hist[-1])
    pseudo_free_energy -= np.min(pseudo_free_energy)
    pseudo_free_energy = pseudo_free_energy.reshape(-1,)
    rewards = pseudo_free_energy*rewards
    return rewards

def CV_correlation_decouple(CV_score_A,CV_score_B,score_cut_off=0.2,correlation_cut_off=0.999):
    arg_A_min = np.where(CV_score_A<score_cut_off)[0]
    arg_B_min = np.where(CV_score_B<score_cut_off)[0]
    n_success_in_A_min = np.where(CV_score_A[arg_B_min]<score_cut_off)[0]
    n_success_in_B_min = np.where(CV_score_B[arg_A_min]<score_cut_off)[0]
    forward_proportion_min = n_success_in_B_min.shape[0]/arg_A_min.shape[0]
    backward_proportion_min = n_success_in_A_min.shape[0]/arg_B_min.shape[0]
    
    arg_A_max = np.where(CV_score_A>score_cut_off)[0]
    arg_B_max = np.where(CV_score_B>score_cut_off)[0]
    n_success_in_A_max = np.where(CV_score_A[arg_B_max]>score_cut_off)[0]
    n_success_in_B_max = np.where(CV_score_B[arg_A_max]>score_cut_off)[0]
    if arg_A_max.shape[0]==0 and arg_B_max.shape[0]>0:
        forward_proportion_max = 1
        backward_proportion_max = 0
    elif arg_B_max.shape[0]==0 and arg_A_max.shape[0]>0:
        forward_proportion_max = 0
        backward_proportion_max = 1
    elif arg_A_max.shape[0]==0 and arg_B_max.shape[0]==0:
        forward_proportion_max = 0
        backward_proportion_max = 0
    else:
        forward_proportion_max = n_success_in_B_max.shape[0]/arg_A_max.shape[0]
        backward_proportion_max = n_success_in_A_max.shape[0]/arg_B_max.shape[0]
    correlation_flag = False
    major_CV = -1 # -1 is no means, 0 represent CV_score_A, while 1 represent CV_score_B
    if (forward_proportion_min>correlation_cut_off) and (backward_proportion_min<0.9) and (backward_proportion_max>correlation_cut_off):
        correlation_flag = True
        major_CV = 0
    elif (backward_proportion_min>correlation_cut_off) and (forward_proportion_min<0.9) and (forward_proportion_max>correlation_cut_off):
        correlation_flag = True
        major_CV = 1
    return correlation_flag, major_CV

def find_root_nodes(data):
    # # Example usage
    # data = [(1,4),(3,7),(2,4),(5,7),(6,7),(4,9),(5,9),(11,15),(9,20)]
    # root_nodes = find_root_nodes(data)
    # print(root_nodes)
    # [15, 20, 7]
    child_set = set()
    father_set = set()

    for child, father in data:
        child_set.add(child)
        father_set.add(father)

    root_nodes = father_set - child_set
    return list(root_nodes)

def find_CV_topology(n_patch,CV_residues_index):
    father_nodes = []
    CV_topology = {}
    n = len(CV_residues_index)
    store_nodes = [[] for i in range(n+n_patch)]
    for i in range(n-1):
        for j in range(i+1,n):
            interset = np.intersect1d(CV_residues_index[i],CV_residues_index[j])
            if interset.shape[0]==len(CV_residues_index[i]) and interset.shape[0]!=len(CV_residues_index[j]):
                if (interset==np.array(CV_residues_index[i])).all():
                    father_nodes.append(j+n_patch)
                    store_nodes[j+n_patch].append(i+n_patch)
            if interset.shape[0]==len(CV_residues_index[j]) and interset.shape[0]!=len(CV_residues_index[i]):
                if  (interset==np.array(CV_residues_index[j])).all():
                    father_nodes.append(i+n_patch)
                    store_nodes[i+n_patch].append(j+n_patch)
    father_nodes = np.unique(father_nodes)
    for father in father_nodes:
        CV_topology[str(father)] = store_nodes[father]
    print('father node:',father_nodes)
    print('CV_topology',CV_topology)
    return father_nodes, CV_topology
 
def filter_CVscore_with_correlation(CV_scores,W,CVgen_score,D_cutoff,n_clusters,candidate_score_scut_off=0.2,method='prior',father_nodes=np.array([]),CV_topology=[]):
    if method == 'auto':
        # parse dynamic topology of CVs
        CV_topology = []
        n_CV = len(CV_scores)
        for i_th in range(n_CV-1):
            for j_th in range(i_th+1,n_CV):
                if np.min(CV_scores[i_th])>=candidate_score_scut_off or np.min(CV_scores[j_th])>=candidate_score_scut_off:
                    continue
                correlation_flag,major_CV = CV_correlation_decouple(CV_scores[i_th],CV_scores[j_th],score_cut_off=candidate_score_scut_off)
                if correlation_flag:
                    child = [i_th,j_th][1-major_CV]
                    father = [i_th,j_th][major_CV]
                    CV_topology.append((child,father))
        print("CV_topology:",CV_topology)
        root_nodes = find_root_nodes(CV_topology)
        true_root_nodes = []
        exclude_child_node = []
        for node in root_nodes:
            nodes_id = np.where(np.array(CV_topology)[:,1]==node)[0]
            father_flag = True
            for pair in np.array(CV_topology)[nodes_id]:
                is_child = np.min(CV_scores[pair[0]])<np.min(CV_scores[pair[1]])
                father_flag = father_flag & is_child
            if nodes_id.shape[0]>1 and father_flag:
                sel_score = CV_scores[node]
                true_root_nodes.append(node)
                exclude_child_node += list(np.array(CV_topology)[nodes_id][:,0])
        print("oringnal root_nodes:",root_nodes)
        print("True root_nodes:",true_root_nodes)
        if len(true_root_nodes)==0:
            CV_indices = np.array([])
            for score in CV_scores:
                CV_indices = np.append(CV_indices,np.where(score<candidate_score_scut_off)[0])
            CV_indices = np.unique(CV_indices.reshape(-1,)).astype(int)
            return CV_indices
        
        # # filter CV_score according to root node of CV topology
        # CV_indices = np.array([])
        # for node in true_root_nodes:
        #     sel_score = CV_scores[node]
        #     CV_indices = np.append(CV_indices,np.where(sel_score<candidate_score_scut_off)[0])
        # CV_indices = np.unique(CV_indices.reshape(-1,)).astype(int)
    elif method == 'prior':
        reached_fathers_arg = np.where(np.min(np.array(CV_scores)[father_nodes],axis=1)<candidate_score_scut_off)[0]
        reached_fathers = father_nodes[reached_fathers_arg]
        exclude_child_node = []
        for father in reached_fathers:
            exclude_child_node += CV_topology[str(father)]
        exclude_child_node = np.unique(exclude_child_node)
        print('reached_fathers:',reached_fathers)
        print('exclude_child_node:',exclude_child_node)

    print('D_cutoff:',D_cutoff)
    percentage_CV = np.array([(np.exp(-i)-np.exp(-1)) for i in W]).reshape(-1,)
    print('percentage_CV:',percentage_CV)
    indices = np.argsort(CV_scores[0])[:int(percentage_CV[0]*CV_scores[0].shape[0])]
    for n,score in enumerate(CV_scores):
        if n in exclude_child_node:
            continue
        # if np.min(score)<candidate_score_scut_off:
        #     indices = np.append(indices,np.where(score<=D_cutoff[n])[0])
        indices = np.intersect1d(indices,np.argsort(score)[:int(percentage_CV[n]*score.shape[0])])
    indices = np.unique(indices.reshape(-1,)).astype(int)
    
    if indices.shape[0]>0:
        bottom_cut = np.max(CVgen_score[indices])
        CV_indices = np.where(CVgen_score<=bottom_cut)[0]
    else:
        CV_indices = np.array([])
    if CV_indices.shape[0]<n_clusters*10:
        CV_indices = np.argsort(CVgen_score)[:n_clusters*10]
    print('cadadite conformations:',CV_indices.shape[0])
    return CV_indices

# def compute_prior_theta(CV_residues_index):
#     all_res = [item for sublist in CV_residues_index for item in sublist]
#     prob_res = {}
#     for res in np.unique(all_res):
#         prob = 1/np.where(np.array(all_res)==res)[0].shape[0]
#         prob_res[str(res)] = prob
    
#     prior_theta = []
#     for residues_index in CV_residues_index:
#         prior_theta.append(np.mean([prob_res[str(res)] for res in residues_index]))
#     prior_theta = np.array(prior_theta)
#     prior_theta = prior_theta/np.sum(prior_theta)
    
#     return prior_theta

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
    delt = 0.6
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
    relative_entropy = np.sum(old_distribution[mask] * np.log2(old_distribution[mask] / new_distribution[mask]))
    
    newarea = combined_data[combined_data<np.min(old_data)]
    if newarea.shape[0]>0:
        cutid1 = np.max(newarea)
        mask1 = bins[:-1]<cutid1
        relative_entropy += np.sum(new_distribution[mask1])
    newarea = combined_data[combined_data>np.max(old_data)]
    if newarea.shape[0]>0:
        cutid2 = np.min(newarea)
        mask2 = bins[1:]>cutid2
        relative_entropy -= np.sum(new_distribution[mask2])

    return bias*relative_entropy

def min_relative_std(x,interval):
    arg_x = np.argmin(x)
    nth_traj = arg_x//interval
    seg_data = x[nth_traj*interval:(nth_traj+1)*interval]
    std = np.std(seg_data)
    relative_std = std/(np.max(x)-np.min(x))
    return relative_std

def optimal_deviation(x,interval,n_replicas):
    arg_x = np.argmin(x)
    nth_traj = arg_x//interval
    seg_data = x[nth_traj*interval:(nth_traj+1)*interval]
    deviation = (np.mean(x)-np.mean(seg_data))/np.std(seg_data)
    return deviation

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
        self.X = X
        self.y = y
        
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

def get_posterior(index_arr, score, interval, max_score=True, mode='extreme'):
    if mode=='extreme':
        if max_score:
            posterior = np.array([np.max(score[i:i+interval]) for i in index_arr])
        else:
            posterior = np.array([np.min(score[i:i+interval]) for i in index_arr])
    elif mode=='mean':
        posterior = np.array([np.mean(score[i:i+interval]) for i in index_arr])
    return posterior

def get_dynamic_prior(score, tica, interval, max_score=True, scale_dynamic_prior='linear'):
    nbins = 500
    distri = cal_prob_dis(tica, n_bins=nbins)
    index_arr = np.arange(score.shape[0])[::interval]

    # Calculate the indices for dynamic prior calculation
    if max_score:
        dynamic_prior_indices = np.array([np.argmax(score[i:i+interval])+i for i in index_arr])
        y = np.array([np.max(score[i:i+interval]) for i in index_arr])
    else:
        dynamic_prior_indices = np.array([np.argmin(score[i:i+interval])+i for i in index_arr])
        y = np.array([np.min(score[i:i+interval]) for i in index_arr])

    # Perform vectorized calculations
    mean_y = np.array([np.mean(score[i:i+interval]) for i in index_arr])
    band_y = np.abs(y - mean_y)
    print('dynamic_prior_indices=',dynamic_prior_indices)
    dynamic_prior_x = tica[dynamic_prior_indices].reshape(-1,)
    dynamic_prior_prob = distri[1][np.abs(distri[0][:, np.newaxis] - dynamic_prior_x).argmin(axis=0).reshape(-1)]
    base = -np.log(np.max(distri[1]))
    print('dynamic_prior_prob=',dynamic_prior_prob)
    pseudo_free_energy_1d = -np.log(dynamic_prior_prob)
    pseudo_free_energy_1d = pseudo_free_energy_1d - base
    print('pseudo_free_energy_1d=',pseudo_free_energy_1d)
    print('base pseudo_free_energy_1d=',base)
    if scale_dynamic_prior=='linear':
        scale_factor = pseudo_free_energy_1d/np.max(pseudo_free_energy_1d)
    elif scale_dynamic_prior=='exponential':
        scale_factor = 1-np.exp(-pseudo_free_energy_1d)
    if max_score:
        dynamic_prior_y = y+scale_factor*band_y
    else:
        dynamic_prior_y = y-scale_factor*band_y
    
    dynamic_prior = [dynamic_prior_x, dynamic_prior_y]
    return dynamic_prior


        
def single_sample_GP_TS(tics_con, interval, posterior, dynamic_prior, num_samples, process_data=None):
        
    x_bounds = (np.min(tics_con), np.max(tics_con))
    x_index = np.arange(0,tics_con.shape[0],interval)
    x = tics_con[x_index]
    x = np.append(x,dynamic_prior[0])
    argsort = np.argsort(x.reshape(-1,))
    y = np.append(posterior,dynamic_prior[1])
    x = x[argsort].reshape(-1,)
    y = y[argsort]
    
    # smoothing data 'y'
    if process_data=='smooth':
        smooth_y = smooth_data(y, window_size=4)
        smooth_y[0] = y[0]
        smooth_y[-1] = y[-1]
        y = smooth_y
    elif process_data=='denoise':
        denoised_y = denoise_data(y, window_size=4, reduction_ratio=1)
        y = denoised_y

    ts_gp = ThompsonSamplingGP(num_samples, x_bounds=x_bounds, X=x, y=y, interval_resolution=2000)
    sampled_x, sampled_y = ts_gp.choose_next_samples()
    out_x = np.abs(tics_con[:, np.newaxis] - sampled_x).argmin(axis=0).reshape(-1,)
    out_data = [x_bounds,x,y,sampled_x,sampled_y]
    with open('GP_data.pkl','ab') as w:
        pickle.dump(out_data, w)
    
    return out_x
    
        
    
def update_multi_paths(old_paths, new_connect_top, interval, posterior):
    """_summary_

    Args:
        old_paths (list):
                        [0] --  a list of 'x' value
                        [1] --  a list of 'y' index
                        [2] --  a list of 'y' value 
                        [3] --  a list of 'y' index in real path
                        [4] --  a list of frame index in corresponding real path interval
        new_connect_top (list):
                        [0] -- 'y' index
                        [1] -- 'frames' number
                        [2] == next 'y' index
     """
    import copy
    flag = False
    copy_y = copy.deepcopy(old_paths[3])
    for n,y_indlst in enumerate(copy_y):
        k = np.where((y_indlst-1)==new_connect_top[0])[0]
        if k.shape[0]>0:
            new_x = np.sum(old_paths[4][n][:k[0]])+new_connect_top[1]
            if (new_x >= old_paths[0][n][-1]) or (new_x in old_paths[0][n]):
                continue
            old_paths[0][n] = np.sort(np.append(old_paths[0][n], new_x.astype(int)))
            arg_x = np.where(old_paths[0][n]==new_x)[0]
            old_paths[1][n] = np.insert(old_paths[1][n],arg_x,new_connect_top[2])
            new_y = posterior[new_connect_top[2]]
            old_paths[2][n] = np.insert(old_paths[2][n],arg_x,new_y)
            if new_connect_top[0] == old_paths[3][n][-1]-1:
                old_paths[3][n] = np.append(old_paths[3][n],new_connect_top[2]+1)
                old_paths[4][n][-1] = new_connect_top[1]
                old_paths[4][n] = np.append(old_paths[4][n],interval)
            elif (new_connect_top[0] != old_paths[3][n][-1]-1) and not flag:
                appd0 = copy.deepcopy(old_paths[0][n][:arg_x[0]+1])
                old_paths[0].append(appd0)
                appd1 = copy.deepcopy(old_paths[1][n][:arg_x[0]+1])
                old_paths[1].append(appd1)
                appd2 = copy.deepcopy(old_paths[2][n][:arg_x[0]+1])
                old_paths[2].append(appd2)
                appd3 = copy.deepcopy(old_paths[3][n][:k[0]+1])
                old_paths[3].append(appd3)
                old_paths[3][-1] = np.append(old_paths[3][-1],new_connect_top[2]+1)
                appd4 = copy.deepcopy(old_paths[4][n][:k[0]+1])
                old_paths[4].append(appd4)
                old_paths[4][-1][-1] = new_connect_top[1]
                old_paths[4][-1] = np.append(old_paths[4][-1],interval)
                flag = True
        else:
            continue
        
    return old_paths

def multi_sample_GP_TS(args):
    old_paths_0, old_paths_2, old_paths_3, old_paths_4, unit_length, num_samples, n_cut, interval = args
    add_samples = 0
    sampled_X = np.array([])
    while sampled_X.shape[0] < num_samples:
        sampled_X = np.array([])
        sampled_Y = np.array([])
        for n,path_x in enumerate(old_paths_0):
            up_bound = (path_x[-1]+interval-1)/unit_length
            X = path_x/unit_length
            y = old_paths_2[n]
            ts_gp = ThompsonSamplingGP(num_samples+add_samples, x_bounds=(0,up_bound), X=X, y=y, interval_resolution=int(up_bound*unit_length/10))
            sampled_x, sampled_y = ts_gp.choose_next_samples()
            scaled_x = np.trunc(sampled_x * unit_length)
            path_index = np.array([np.sum(old_paths_4[n][:des+1]) for des in range(old_paths_4[n].shape[0])])
            arg_x = np.array([np.min(np.where(path_index>=i)[0]) for i in scaled_x])
            if arg_x[0]==0:
                remain_frames = scaled_x
                sampled_gindex = remain_frames
            else:
                remain_frames = scaled_x - path_index[arg_x-1]
                sampled_gindex = old_paths_3[n][arg_x-1] * interval + remain_frames
            sampled_X = np.append(sampled_X, sampled_gindex)
            sampled_Y = np.append(sampled_Y, sampled_y)
        
        if sampled_X.shape[0]>n_cut:
            break
        # exclude same x
        remove_indices = []
        for i in np.unique(sampled_X):
            indices = np.where(sampled_X == i)[0]
            min_index = indices[np.argmin(sampled_Y[indices])]
            remove_indices.extend(indices[indices != min_index])

        sampled_X = np.delete(sampled_X, remove_indices)
        sampled_Y = np.delete(sampled_Y, remove_indices)
        
        add_samples += 5
        print("add_samples=",add_samples)
        del ts_gp
        
    
    return sampled_X.astype(int), sampled_Y

def get_tica(ndim,lagtime):
    if(os.path.isfile('data_output.pkl')):
        with open('data_output.pkl','rb') as in_data:
            data_output = pickle.load(in_data)

    tica_estimator = TICA(lagtime=lagtime, dim=ndim)
    tica = tica_estimator.fit(data_output).fetch_model()
    print('TICA dimension = ', tica.output_dimension)
    tics = tica.transform(data_output)
    return tics
        
def random_select_indices(arr, N):
    indices = np.arange(len(arr))
    np.random.shuffle(indices)
    sel_indices = indices[:N]
    sel_values = arr[sel_indices]
    return sel_values, sel_indices

def choose_bandit(arms_values, dtrajs, N_per_arm, N_arms, N_tot_actions, max_score=True):
    # list of samples, for each bandit
    samples_list = []
                
    # drawing a sample from each bandit distribution
    for values in arms_values:
        sampled_values = np.mean(random.choices(values, k=N_per_arm))
        indices = np.where(values == sampled_values[:, None])[1]
        samples_list.append(sampled_values)
    samples_list = np.array([samples_list])
    
    if max_score:  
        index = np.argsort(samples_list)[-N_arms:]
    else:
        index = np.argsort(samples_list)[:N_arms]
    dtrajs_con = np.concatenate(dtrajs)
    filtered_index = [np.where(dtrajs_con==i)[0] for i in index]
    samples_index = []
    for indices in filtered_index:
        samples_index += list(np.random.choice(indices, int(N_tot_actions/N_arms)))
                    
    # returning bandit with best sample
    return samples_index  

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

    while True:
        try:
            cluster = KMeans(n_clusters, max_iter=200).fit_fetch(np.concatenate(tics)[::10])
            dtrajs = [cluster.transform(traj) for traj in tics]
            dtrajs_concatenated = np.concatenate(dtrajs)
            print('dtraj index start from ', np.min(dtrajs))
            counts_estimator = TransitionCountEstimator(lagtime=lagtime, count_mode='sliding')
            counts = counts_estimator.fit_fetch(dtrajs).submodel_largest()
            msm_estimator = MaximumLikelihoodMSM()
            msm = msm_estimator.fit_fetch(counts)
            break
        except Exception as e:
            n_clusters -= 1
            print("Error occurred:", str(e))
            print("Exception handling,n_clusters=",n_clusters)
            
    stat_dis = msm.stationary_distribution
        
    arms_values = []
    arms_indices = []
    if prior_score.shape[0] != 0:
        for i in range(n_clusters):
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

        for i in range(n_clusters):
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

def cal_prob_dis(x, n_bins=500):
    # Calculate the histogram
    hist, bins = np.histogram(x, bins=n_bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    return [bin_centers, hist]

def smooth_data(data, window_size):
    # Define the smoothing window
    window = np.ones(window_size) / window_size
    
    # Apply convolution to smooth the data
    smoothed_data = np.convolve(data, window, mode='same')
    
    return smoothed_data

def denoise_data(data, window_size=4, reduction_ratio=1):
    from scipy.signal import argrelextrema
    # Find local minima and maxima indices
    minima_indices = argrelextrema(data, np.less)[0]
    maxima_indices = argrelextrema(data, np.greater)[0]

    # Reduce maximum values of all peaks proportionally to the distance from local minima
    denoised_data = np.copy(data)

    for max_idx in maxima_indices:
        closest_min_idx = minima_indices[np.argsort(np.abs(max_idx - minima_indices))[:window_size]]
        distance = np.max(np.abs(data[max_idx] - data[closest_min_idx]))
        reduction_factor = distance * reduction_ratio
        denoised_data[max_idx] -= reduction_factor
        
    return denoised_data

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

def feature(dirs, pdb):
    feat = pyemma.coordinates.featurizer(pdb)
    pairs = feat.pairs(feat.select_Ca())
    feat.add_contacts(pairs,threshold=0.8)
    data = pyemma.coordinates.source(dirs, features=feat)
    data_output = data.get_output()
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

def read_whole_traj(file_dirs_list, pdb):
    trajs = []
    data_output = []
    for i in file_dirs_list:
        file_list = os.listdir(i)
        if(len(file_list) != 0):
            num_round = len(file_list)
            whole_dirs = [i+'/'+str(j)+'.dcd' for j in range(num_round)]
            output = feature(whole_dirs, pdb)
            output_con = []
            for data in output:
                inter = args.sampling_interval
                n = data.shape[0]//inter
                for l in range(n):
                    output_con.append(data[l*inter:(l+1)*inter])
            for data in output_con:
                data_output.append(data)
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

def read_slice_traj(file_dir, pdb, nth_round):
    interval = int(-1 * args.sampling_interval)
    trajs = []
    file_list = []
    for i in file_dir:
        ifile=os.listdir(i)
        if len(ifile)==0:
            continue
        file_list.append(i+'/'+str(nth_round)+'.dcd')
    if(len(file_list)==0):
        whole_traj = None
        return whole_traj
    updated_data_output = feature(file_list, pdb)

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

def cal_rmsd_score(args):
    traj, ref, heavy = args
    rmsd = md.rmsd(traj, reference=ref, atom_indices=heavy)

    return rmsd

    
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
    traj_slice,pdb = args
    ref_contact_short, ref_contact_long = analyze_contact(pdb)
    contacts_short = md.compute_contacts(traj_slice, contacts=ref_contact_short[1], scheme='closest-heavy')
    contacts_long = md.compute_contacts(traj_slice, contacts=ref_contact_long[1], scheme='closest-heavy')
    scores_short = contact_score(contacts_short,ref_contact_short,0.05)
    scores_long = contact_score(contacts_long,ref_contact_long,0.05)
    return scores_short, scores_long
    
    
def cal_dssp_score(args):

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

    pdb = args.pdb
    ref = md.load_pdb(pdb)
    heavy = ref.top.select_atom_indices('heavy')

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
            hbonds_lst = md.baker_hubbard(ref,distance_cutoff=0.25)
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
        whole_traj = read_whole_traj(whole_dirs, pdb)
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

def allocate_elements(CV_weights, N):
    total_weights = np.sum(CV_weights)
    normalized_weights = CV_weights / total_weights

    allocated_counts = np.round(normalized_weights * N).astype(int)
    remaining_counts = N - np.sum(allocated_counts)

    if remaining_counts > 0:
        sorted_indices = np.argsort(CV_weights)[::-1][:remaining_counts]
        allocated_counts[sorted_indices] += 1

    return allocated_counts

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
    
# def left_skew(data):
#     from scipy.stats import skew
#     mean = np.mean(data)
#     left_data = data[data <= mean]
#     left_skewness = skew(left_data, bias=False)
#     return left_skewness

def optimize_ALPHA(CV_residues_index,theta,cache_scores,n_replicas,interval,delt):
    def banlanced_score(whole_data,selected_index,weights):
        exploration_score = []
        exploitation_score = []
        start = selected_index*interval
        end = (selected_index+1)*interval
        for data in whole_data:
            arg_data = np.argmin(data)
            nth_traj = arg_data//interval
            new_data = data[nth_traj*interval:(nth_traj+1)*interval]
                
            sel_data = data[start:end]
            exploration_score.append(np.mean(data)-np.mean(sel_data))
            # exploitation_score.append(np.min(data)-np.mean(sel_data))
            exploitation_score.append(np.mean(new_data)-np.mean(sel_data))
        exploration_score = np.array(exploration_score)
        exploitation_score = np.array(exploitation_score)
        weights = weights.reshape(1,-1)
        EE_score = np.dot(weights,(exploration_score+exploitation_score).reshape(-1,1))
        return EE_score
    
    # if np.max(delt)<=0:
    #     return 0.5
    # all_res = np.array([item for sublist in CV_residues_index for item in sublist])
    # unique_res = np.unique(all_res)
    # prior_weight = np.array([-np.log(len(ith_CV)/unique_res.shape[0]) for ith_CV in CV_residues_index])
    # prior_weight = prior_weight/np.sum(prior_weight)
    prior_weight = theta
    
    default_resolution = 500
    theta = np.array(theta).reshape(-1,1)
    delt = np.array(delt).reshape(-1,1)
    if np.max(np.abs(delt))==0:
        return 0
    d = 10/np.max(np.abs(np.dot(theta.reshape(1,-1),delt)))
    low_bound_ratio = 0.01
    step = ((1-low_bound_ratio)*d)/(default_resolution)
    alpha = np.array([np.arange(low_bound_ratio*d,d,step) for _ in range(len(cache_scores))])
    print('alpha_min=',np.min(alpha[0]),'alpha_max=',np.max(alpha[0]),'alpha.shape=',alpha.shape)
    W = np.exp(alpha*delt) # W.shape = (N,default_resolution), N represents number of CVs
    W = W*theta
    W = W.T
    W = W/np.sum(W,axis=1).reshape(-1,1)
    R = np.dot(W,np.array(cache_scores)) # R.shape = (default_resolution,M), M represents number of samples
    
    related_intervals = np.unique(np.concatenate(np.argsort(R,axis=1)[:,:interval])//interval)
    EE_scores = np.array([banlanced_score(cache_scores,selected_index,prior_weight) for selected_index in related_intervals])
    optimized_scores = []
    for r in R:
        all_intervals = np.argsort(r)[:interval]//interval
        rela_intervals = np.unique(all_intervals)
        ratios = np.array([np.where(all_intervals==i)[0].shape[0]/all_intervals.shape[0] for i in rela_intervals]).reshape(1,-1)
        ee_scores = np.array([EE_scores[np.where(related_intervals==i)[0]] for i in rela_intervals]).reshape(-1,1)
        optimized_scores.append(np.dot(ratios,ee_scores))
    
    print('Part of optimized score :',optimized_scores[::10])
    optimized_ALPHA = alpha[0][np.argmax(optimized_scores)]
    return optimized_ALPHA
    
def cal_loss(theta,cache_scores,number_replica,interval,k=1,delt=None,method="relative_entropy",cluster_labels=None):
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
        theta = np.array(theta).reshape(-1,1)
        n_clusters = np.unique(cluster_labels).shape[0]
        
        default_resolution = 50
        # delt = [min_relative_std(x=score,interval=interval) for score in cache_scores]
        delt = np.array(delt).reshape(-1,1)
        if np.max(delt)<=0:
            return np.array([0]), np.array([0])
        d = 10/np.max(delt)
        step = d/(default_resolution)
        # common_ratio = 0.9
        # alpha = np.array([[d*common_ratio**i for i in range(default_resolution)] for _ in range(len(cache_scores))])
        alpha = np.array([np.arange(0,d,step) for _ in range(len(cache_scores))])
        print('alpha_min=',np.min(alpha[0]),'alpha_max=',np.max(alpha[0]),'alpha.shape=',alpha.shape)
        
        W = np.exp(alpha*delt) # W.shape = (N,default_resolution), N represents number of CVs
        W = W*theta
        W = W.T
        W = W/np.sum(W,axis=1).reshape(-1,1)
        
        D_KL = []
        for i in range(n_clusters):
            d_kl = np.array([directional_relative_entropy(old_data=score[-number_replica*interval:],
                                                           new_data=score[cluster_labels==i])
                             for score in cache_scores]).reshape(1,-1)
            # d_kl = np.array([1/np.std(score[cluster_labels==i])
            #                  for score in cache_scores]).reshape(1,-1)
            D_KL.append(np.dot(d_kl,W.T))
        D_KL = np.array(D_KL).reshape(n_clusters,default_resolution)
        D_KL = D_KL-np.min(D_KL,axis=0).reshape(1,-1)
        D_KL = D_KL/np.sum(D_KL,axis=0).reshape(1,-1)
        
        R = np.dot(W,np.array(cache_scores)) # R.shape = (default_resolution,M), M represents number of samples
        R = R/np.sum(R,axis=1).reshape(-1,1)
        CG_R = np.zeros((default_resolution,n_clusters)) # CG_R.shape = (default_resolution,n_clusters)
        for i in range(n_clusters):
            indices = cluster_labels==i
            CG_R[:,i] = np.mean(R[:,indices],axis=1)
        CG_R = CG_R.T # CG_R.shape = (n_clusters,default_resolution)
        cumulant_reward = np.sum(D_KL*CG_R,axis=0)
        print("CG_R:",CG_R)
        print("D_KL:",D_KL)
        print("cumulant_reward = ",cumulant_reward)
        optimal_ALPHA = alpha[0][np.argmin(cumulant_reward)]
        print("optimal_ALPHA = ",optimal_ALPHA)
        sampled_alpha = np.array([optimal_ALPHA])
        D_KL = np.array([optimal_ALPHA])
        # print('R.shape=',R.shape)
        # samples_index = np.argmin(R,axis=1)
        # samples_index = np.unique(samples_index)
        # print('samples_index=',samples_index)
        # seed_arr = np.arange(0,cache_scores[0].shape[0],interval)
        # sampled_alpha = []
        # D_KL = []
        # for n,i in enumerate(seed_arr): # this loop can be improved
        #     mask = np.where(samples_index>=i)[0]
        #     if mask.shape[0]==0:
        #         continue
        #     index = mask[np.argmin(samples_index[mask]-i)]
        #     if samples_index[index]-i<interval:
        #         sampled_alpha.append(alpha[0][index])
        #         d_kl = np.array([-directional_relative_entropy(old_data=score[-number_replica*interval:],
        #                                              new_data=score[n*interval:(n+1)*interval])
        #                 for score in cache_scores]).reshape(1,-1)
                
        #         D_KL.append(np.dot(d_kl,theta))
        # sampled_alpha = np.array(sampled_alpha)
        # D_KL = np.array(D_KL).reshape(-1,)
        # print("sampled_alpha:",sampled_alpha,"D_KL:",D_KL)
        
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
    latent_score = score + uncertainty

    return latent_score

def compute_decision_cutoff(data):
    from scipy.stats import gaussian_kde
    # Estimate the kernel density of the data
    kde = gaussian_kde(data, bw_method=0.5)
    # Create a smooth curve using the estimated kernel density
    x_ = np.linspace(min(data), max(data), 200)
    y_ = kde(x_)
    y_ /= np.max(y_)

    # Calculate the derivatives of the fitted curve
    dx = x_[1] - x_[0]
    dy_dx = np.gradient(y_, dx)
    # Find the point on the left side with maximum derivative
    idx_max_derivative = np.argmax(dy_dx[:len(dy_dx)//2])  # Limit search to left side
    point_max_derivative = x_[idx_max_derivative]
    
    return point_max_derivative

def unbiased_distribution_index(data):
    min_val = np.min(data)
    max_val = np.max(data)

    for N in range(2, len(data)):
        segment_size = (max_val - min_val) / N
        hist, bins = np.histogram(data, bins=N, range=(min_val, max_val))
        if np.any(hist == 0):
            break
        selected_indices = []
        for i in range(N):
            bin_indices = np.where((data >= bins[i]) & (data <= bins[i+1]))[0]
            selected_index = np.random.choice(bin_indices)
            selected_indices.append(selected_index)

    return np.array(selected_indices)

def folding_constrain_factor(args):
    traj, ref = args
    
    cutoff = 2
    import itertools
    n_residues = ref.topology.n_residues
    all_resi = np.arange(n_residues)
    flag_alpha = False
    flag_beta = False
    ref_dssp_code = md.compute_dssp(ref, simplified=True)
    alpha_res_index = np.where(ref_dssp_code[0]=='H')[0]
    beta_res_index =  np.where(ref_dssp_code[0]=='E')[0]
    if alpha_res_index.shape[0]>0:
        alpha_res = extract_consecutive_parts(alpha_res_index)
        flag_alpha = True
    if beta_res_index.shape[0]>0:
        beta_res = extract_consecutive_parts(beta_res_index)
        flag_beta = True
        
    alpha_constrain_factor = np.zeros(len(traj))
    if flag_alpha:
        cons_alpha = np.array([])
        for frag in alpha_res:
            frag_ = np.arange(np.min(frag)-2,np.max(frag)+2)
            other_res = np.setdiff1d(all_resi,frag_,True)
            pairs = list(itertools.product(frag, other_res))
            contact_distance = md.compute_contacts(traj=traj,contacts=pairs,scheme='closest')[0]
            # print("contact_distance:",contact_distance)
            contact_distance[contact_distance > cutoff] = cutoff
            cons_alpha = np.append(cons_alpha,np.sum(contact_distance,axis=1))
        cons_alpha = cons_alpha.reshape(-1,len(traj))
        alpha_constrain_factor += np.sum(cons_alpha,axis=0)
    
    beta_constrain_factor = np.zeros(len(traj))
    if flag_beta:
        # Create a list of coarse-grained hydrogen bonds represented by pairs of amino acids.
        res_hbonds = []
        hbonds_lst = md.baker_hubbard(ref,distance_cutoff=0.25)
        res2atom = []
        for i in range(n_residues):
            at1 = ref.top.select('resid '+str(i))[0]
            at2 = ref.top.select('resid '+str(i))[-1]
            res2atom.append([at1,at2])
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
            if sum(np.isin(np.concatenate(beta_res), hbond[0]))\
                and sum(np.isin(np.concatenate(beta_res), hbond[1])):
                res_hbonds_beta.append(hbond)
        
        beta_pair = []
        for pair in res_hbonds_beta:
            ind = [-1, -1]
            for n,lst in enumerate(beta_res):
                if sum(np.isin(lst,pair[0])):
                    ind[0] = n
                elif sum(np.isin(lst,pair[1])):
                    ind[1] = n
            beta_pair.append(np.sort(np.array(ind)))
        beta_pair = np.unique(beta_pair, axis=0)
        direct_b_pair = []
        for p in beta_pair:
            start = beta_res[p[0]][-1]+1
            end = beta_res[p[1]][0]
            if sum(ref_dssp_code[0][start:end]=='H') or sum(ref_dssp_code[0][start:end]=='E'):
                continue
            else:
                direct_b_pair.append(p)
        
        cons_beta = np.array([])
        if len(direct_b_pair)>0:
            for pair in direct_b_pair:
                frag = np.concatenate([beta_res[pair[0]],beta_res[pair[1]]])
                frag_ = np.arange(np.min(frag)-2,np.max(frag)+2)
                other_res = np.setdiff1d(all_resi,frag_,True)
                pairs = list(itertools.product(frag, other_res))
                contact_distance = md.compute_contacts(traj=traj,contacts=pairs,scheme='closest')[0]
                contact_distance[contact_distance > cutoff] = cutoff
                cons_beta = np.append(cons_beta,np.sum(contact_distance,axis=1))
            cons_beta = cons_beta.reshape(-1,len(traj))
            beta_constrain_factor += np.sum(cons_beta,axis=0)
    
    constrain_factor = alpha_constrain_factor + beta_constrain_factor
    return constrain_factor

def hydrophobic_score_factor(args):
    import itertools
    pdb,traj,res_index,CV_residues_index = args
    cutoff = 0.45
    unit_factor = 10 # 0.1nm
    ref = md.load_pdb(pdb)
    ref_dssp_code = md.compute_dssp(ref, simplified=False)[0]
    structed_res = np.where((ref_dssp_code=='H') | (ref_dssp_code=='E'))[0]
    structed_domain = extract_consecutive_parts(structed_res)
    structed_domain_indices = []
    for domain in structed_domain:
        structed_domain_indices.append(ref.topology.select('resi '+' '.join(str(i) for i in domain)))
    
    structed_atoms = ref.topology.select('resi '+' '.join(str(i) for i in structed_res))
    structed_res_index = np.intersect1d(structed_res,res_index)
    
    indices_lst = [ref.topology.select('resi '+str(i)) for i in res_index]
    resi_indices = ref.topology.select('resi '+' '.join(str(i) for i in res_index)) # All atoms corresponding to hydrophobic residues
    heavy = ref.top.select_atom_indices('heavy')
    sidechain = ref.top.select('sidechain')
    heavy_sidechain = np.intersect1d(heavy,sidechain)
    hydrophobic_indices = np.intersect1d(heavy_sidechain,resi_indices) # All side chain atoms corresponding to hydrophobic residues
    tmp_lst = [np.intersect1d(hydrophobic_indices,indices) for indices in indices_lst]
    reduced_lst = [np.intersect1d(structed_atoms,indices) for indices in tmp_lst]
    exclude_domain = [np.array([])]*len(reduced_lst)
    for n,indice in enumerate(reduced_lst):
        for domain in structed_domain_indices:
            if np.intersect1d(indice,domain).shape[0]>0:
                exclude_domain[n] = domain
                break
            
                
        
    # print('Structural hydrophobic residue side chains :',reduced_lst)
    structed_heavy = np.intersect1d(heavy,structed_atoms)
    
    # identify the hydrophobic cores positions
    target_hydrophobic_position = []
    target_interact_pairs = []
    exclude_res = []
    for n,indices in enumerate(reduced_lst):
        if indices.shape[0]==0:
            continue
        other_indices = np.setdiff1d(structed_heavy,exclude_domain[n])
        pairs = np.array(list(itertools.product(indices, np.array(other_indices))))
        pairs_distances = md.compute_distances(ref,pairs).reshape(-1,)
        pair_index = np.where(pairs_distances<cutoff)[0]
        if pair_index.shape[0]==0:
            exclude_res.append(res_index[n])
            continue
        key_pair_distances = pairs_distances[pair_index]
        interact_pairs = pairs[pair_index]
        target_hydrophobic_position.append(key_pair_distances)
        target_interact_pairs.append(interact_pairs)
        # print('hydrophobic core',n,':',interact_pairs)
    structed_res_index = np.setdiff1d(structed_res_index,exclude_res)
    
    hydrophobic_cores_deviation = []
    for n,pairs in enumerate(target_interact_pairs):
        pairs_distances = md.compute_distances(traj,pairs)
        deviations = unit_factor*np.mean(np.abs(pairs_distances-target_hydrophobic_position[n]),axis=1).reshape(-1,)
        hydrophobic_cores_deviation.append(deviations)
    hydrophobic_cores_deviation = np.array(hydrophobic_cores_deviation) #  shape=(number_of_hydrophobic_cores,n_frames)
    
    # compute hydrophobic_score_factor for each cv
    score_factors = []
    for cv_res in CV_residues_index:
        sel_res = np.intersect1d(cv_res,structed_res_index)
        if sel_res.shape[0]==0:
            score_factors.append(np.zeros(len(traj)))
        else:
            sel_index = np.array([np.where(structed_res_index==i)[0] for i in sel_res])
            score_factors.append(np.mean(hydrophobic_cores_deviation[sel_index],axis=0).reshape(-1,))
    # print(score_factors)
    score_factors = np.array(score_factors)
    
    return score_factors
    
    
    
def hydrophobic_score_residue(native_pdb_name):
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP
    import warnings

    amino_acid_map = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLN": "Q",
            "GLU": "E",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V"
        }

    kyte_doolittle_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
            'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
            'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Step 1: Parse the PDB file
        parser = PDBParser()
        structure = parser.get_structure("protein", native_pdb_name)

        # Step 2: Calculate solvent accessibility
        model = structure[0]
        dssp = DSSP(model, native_pdb_name)
    solvent_accessibility = {dssp_residue[0]: dssp_residue[3] for dssp_residue in dssp}

    # Step 3: Calculate hydrophobicity using Kyte-Doolittle scale
    protein_sequence = list(structure[0].get_residues())
    protein_sequence = [amino_acid_map[residue.get_resname()] for residue in protein_sequence]
    # print(protein_sequence)
    protein_sequence = ''.join(protein_sequence)
    hydrophobicity = [kyte_doolittle_scale[i] for i in protein_sequence]

    # Step 4: Identify residues in the hydrophobic core
    hydrophobic_core = []
    for n,residue in enumerate(structure.get_residues()):
        residue_id = residue.get_full_id()
        residue_number = residue_id[3][1]
        residue_chain = residue_id[2]
        residue_key = (residue_chain, residue_number)
        # if residue_key in solvent_accessibility and residue_key in hydrophobicity:
        if residue_number in solvent_accessibility:
            accessibility = solvent_accessibility.get(residue_number)
            hydrophobic_score = hydrophobicity[n]
            if accessibility < 0.4 and hydrophobic_score > 0:
                hydrophobic_core.append(residue_number)
                
    return np.array(hydrophobic_core)-1

def is_pre_folding_stage(traj, ref):    
    flag = 1
    ref_dssp_code = md.compute_dssp(ref, simplified=True)
    alpha_res_index = np.where(ref_dssp_code[0]=='H')[0]
    beta_res_index =  np.where(ref_dssp_code[0]=='E')[0]
    
    dssp_code = md.compute_dssp(traj,simplified=True)
    alpha_score = []
    beta_score = []
    for code in dssp_code:
        if alpha_res_index.shape[0]>0:
            alpha_score.append(np.sum((ref_dssp_code[0]==code) & (code!='E') & (code!='C'))/alpha_res_index.shape[0])
        if beta_res_index.shape[0]>0:
            beta_score.append(np.sum((ref_dssp_code[0]==code) & (code!='H') & (code!='C'))/beta_res_index.shape[0])
    if alpha_res_index.shape[0]>0 and beta_res_index.shape[0]>0:
        if np.max(alpha_score)>0.9 and np.max(beta_score)>0.6:
            flag = 0
    elif alpha_res_index.shape[0]>0 and beta_res_index.shape[0]==0:
        if np.max(alpha_score)>0.9:
            flag = 0
    elif alpha_res_index.shape[0]==0 and beta_res_index.shape[0]>0:
        if np.max(beta_score)>0.6:
            flag = 0
            
    return flag

def filter_folding_index(traj, ref, least_n_strucs, simplified=True):
    def transfer_dssp_code(dssp_code):
        helix_like_index = (dssp_code=='G')+(dssp_code=='H')+(dssp_code=='I')+(dssp_code=='T')
        sheet_like_index = (dssp_code=='E')+(dssp_code=='B')
        dssp_code[helix_like_index] = 'X'
        dssp_code[sheet_like_index] = 'Y'
        return dssp_code
    
    history_dssp_score = np.array([])
    if os.path.isfile('history_dssp_score.npy'):
        with open('history_dssp_score.npy','rb') as r:
            history_dssp_score = np.load(r)
            
    ref_dssp_code = md.compute_dssp(ref, simplified=simplified)[0]
    dssp_code = md.compute_dssp(traj,simplified=simplified)
    # if not simplified:
    #     ref_dssp_code = transfer_dssp_code(ref_dssp_code)
    #     dssp_code = transfer_dssp_code(dssp_code)

    # reward_score = np.array([np.sum(ref_dssp_code==code)/ref_dssp_code.shape[0] for code in dssp_code])
    # punish_score = np.array([-np.sum(ref_dssp_code!=code)/ref_dssp_code.shape[0] for code in dssp_code])
    # dssp_score = reward_score+punish_score
    dssp_score = np.array([np.sum((ref_dssp_code==code) & (ref_dssp_code!='C'))/ref_dssp_code.shape[0] for code in dssp_code])
    
    history_dssp_score = np.append(history_dssp_score,dssp_score)
    with open('history_dssp_score.npy','wb') as w:
        np.save(w,history_dssp_score)
            
    n=0
    step = (np.max(history_dssp_score)-np.unique(history_dssp_score)[-2])/10
    filtered_index = np.array([])
    max_dssp = np.max(history_dssp_score)
    while filtered_index.shape[0]<least_n_strucs:
        filtered_index = np.where(history_dssp_score>=(max_dssp-n))[0]
        n += step
            
    return filtered_index

def filter_post_folding_index(traj, ref, least_n_strucs):
    ref_dssp_code = md.compute_dssp(ref, simplified=True)
    alpha_res_index = np.where(ref_dssp_code[0]=='H')[0]
    beta_res_index =  np.where(ref_dssp_code[0]=='E')[0]
    
    dssp_code = md.compute_dssp(traj,simplified=True)
    alpha_score = []
    beta_score = []
    for code in dssp_code:
        if alpha_res_index.shape[0]>0:
            alpha_score.append(np.sum((ref_dssp_code[0]==code) & (code!='E') & (code!='C'))/alpha_res_index.shape[0])
        if beta_res_index.shape[0]>0:
            beta_score.append(np.sum((ref_dssp_code[0]==code) & (code!='H') & (code!='C'))/beta_res_index.shape[0])
    
    n=0
    step = 0.0005
    filtered_index = np.array([])
    if alpha_res_index.shape[0]>0 and beta_res_index.shape[0]>0:
        while filtered_index.shape[0]<least_n_strucs:
            index_alpha = np.where(alpha_score>(0.9-n))[0]
            index_beta = np.where(beta_score>(0.9-n))[0]
            filtered_index = np.intersect1d(index_alpha,index_beta)
            n += step
    elif alpha_res_index.shape[0]>0 and beta_res_index.shape[0]==0:
        while filtered_index.shape[0]<least_n_strucs:
            filtered_index = np.where(alpha_score>(0.9-n))[0]
            n += step
    elif alpha_res_index.shape[0]==0 and beta_res_index.shape[0]>0:
        while filtered_index.shape[0]<least_n_strucs:
            filtered_index = np.where(beta_score>(0.9-n))[0]
            n += step
            
    return filtered_index

def is_post_folding_CV(ref,CV_index):
    ref_dssp_code = md.compute_dssp(ref, simplified=True)[0]
    
    num_CV = len(CV_index)
    post_folding_CV = np.zeros(num_CV)
    for n,cv in enumerate(CV_index):
        cv_res = extract_consecutive_parts(cv)
        if len(cv_res)==2:
            if np.min(cv_res[1])>np.max(cv_res[0]):
                inter_part = np.arange(np.max(cv_res[0])+1,np.min(cv_res[1]))
                if np.sum(ref_dssp_code[inter_part]!='C')>0:
                    post_folding_CV[n] = 1
            elif np.min(cv_res[0])>np.max(cv_res[1]):
                inter_part = np.arange(np.max(cv_res[1])+1,np.min(cv_res[0]))
                if np.sum(ref_dssp_code[inter_part]!='C')>0:
                    post_folding_CV[n] = 1
    
    return post_folding_CV.astype(int)

def compute_current_freedoms(args):
    traj,ref,CV_index,metrix = args
    
    def transfer_dssp_code(dssp_code):
        helix_like_index = (dssp_code=='G')+(dssp_code=='H')+(dssp_code=='I')+(dssp_code=='T')
        sheet_like_index = (dssp_code=='E')+(dssp_code=='B')
        dssp_code[helix_like_index] = 'X'
        dssp_code[sheet_like_index] = 'Y'
        return dssp_code
    
    def arr2set(arr):
        out_set = np.concatenate((arr[:,0].reshape(-1,1),arr[:,2].reshape(-1,1)),axis=1)
        out_set = np.append(out_set,np.concatenate((arr[:,2].reshape(-1,1),arr[:,0].reshape(-1,1)),axis=1),axis=0)
        out_set = set(map(tuple, out_set))
        return out_set
    
    if metrix=='hbond':
        n_atoms = ref.n_atoms
        n_residues = ref.n_residues
        # print('cv',n,'cv_atoms=',n_atoms,'cv_residues=',n_residues)
        atom2res = np.arange(n_atoms)
        for i_res in range(n_residues):
            indices = ref.topology.select('resi '+str(i_res))
            atom2res[indices] = i_res
            
        # CV_atom2res = []
        # for n,cv in enumerate(CV_index):
        #     cv_atoms = ref.topology.select('resi '+' '.join(str(i) for i in cv))
        #     cv_ref = ref.atom_slice(cv_atoms)
            
        #     n_atoms = cv_ref.n_atoms
        #     n_residues = cv_ref.n_residues
        #     # print('cv',n,'cv_atoms=',n_atoms,'cv_residues=',n_residues)
        #     atom2res = np.arange(n_atoms)
        #     for i_res in range(n_residues):
        #         indices = cv_ref.topology.select('resi '+str(i_res))
        #         atom2res[indices] = i_res
        #     CV_atom2res.append(atom2res)
            
        ref_h_bond = md.baker_hubbard(ref,distance_cutoff=0.25)
        h_indices = ref_h_bond.flatten()
        ref_hbond_res = atom2res[h_indices]
        
        h_bonds_sets = [arr2set(md.baker_hubbard(frame,distance_cutoff=0.25)) for frame in traj]
        ref_h_set = np.concatenate((ref_h_bond[:,0].reshape(-1,1),ref_h_bond[:,2].reshape(-1,1)),axis=1)
        ref_h_set = set(map(tuple, ref_h_set))
        non_hbonds = [ref_h_set-hbond_set for hbond_set in h_bonds_sets]
        non_hbonds_atom_index = [np.array(list(non_hbond)).flatten() for non_hbond in non_hbonds]
        non_hbonds_res_index = [atom2res[atom_index] for atom_index in non_hbonds_atom_index]
        CV_DOF = [] # degree of freedoms of each CV
        for cv in CV_index:
            cv_atoms = ref.topology.select('resi '+' '.join(str(i) for i in cv))        
            
            cv_res = atom2res[cv_atoms]
            free_res = np.setdiff1d(cv_res,ref_hbond_res)
            if free_res.shape[0]==cv_res.shape[0]:
                CV_DOF.append(len(cv))
                continue
            # print('before free_res',n,'=',free_res)
            
            free_ress = [np.union1d(free_res,np.intersect1d(cv_res,non_hbond_res)) for non_hbond_res in non_hbonds_res_index]
            len_free_ress = [length.shape[0] for length in free_ress]
            CV_DOF.append(np.min(len_free_ress))
            
    elif metrix=='SS':
        simplified = False
        CV_DOF = [] # degree of freedoms of each CV
        ref_dssp_code = md.compute_dssp(ref, simplified=simplified)[0]
        dssp_code = md.compute_dssp(traj,simplified=simplified)
        # R_record = np.array([False]*ref_dssp_code.shape[0])
        # hbond_dssp_index = (ref_dssp_code!=' ') & (ref_dssp_code!='S')
        # records = np.array([(ref_dssp_code[hbond_dssp_index]==code[hbond_dssp_index]) for code in dssp_code])
        # R = np.sum(records,axis=0)
        # R_record[hbond_dssp_index] = R
        
        # if os.path.isfile('res_record.npy'):
        #     with open('res_record.npy','rb') as r:
        #         res_record = np.load(r)
        #     R_record = np.sum(np.array([R_record,res_record]),axis=0)
        # with open('res_record.npy','wb') as w:
        #     np.save(w,R_record)
        
        # if not simplified:
        #     ref_dssp_code = transfer_dssp_code(ref_dssp_code)
        #     dssp_code = transfer_dssp_code(dssp_code)
            
        for cv in CV_index:
            free_res = np.sum(ref_dssp_code[np.array(cv)]==' ') + np.sum(ref_dssp_code[np.array(cv)]=='S')
            free_ress = [np.sum((ref_dssp_code[np.array(cv)]!=code[np.array(cv)]) & (ref_dssp_code[np.array(cv)]!=' ') & (ref_dssp_code[np.array(cv)]!='S')) for code in dssp_code]
            free_res += np.min(free_ress)
            CV_DOF.append(free_res)
        
    return CV_DOF

def compute_traj_frame_freedoms(traj,ref,CV_index):
    simplified = False
    CV_DOF = [] # degree of freedoms of each CV
    ref_dssp_code = md.compute_dssp(ref, simplified=simplified)[0]
    dssp_code = md.compute_dssp(traj,simplified=simplified)
    for cv in CV_index:
        free_res = np.sum((ref_dssp_code[np.array(cv)]==' ') | (ref_dssp_code[np.array(cv)]=='S'))
        free_ress = [np.sum((ref_dssp_code[np.array(cv)]!=code[np.array(cv)]) & (ref_dssp_code[np.array(cv)]!=' ') & (ref_dssp_code[np.array(cv)]!='S')) for code in dssp_code]
        free_res = np.array(free_ress)+free_res
        CV_DOF.append(free_res)
        
    return np.array(CV_DOF).T

def compute_cv_free_energy(pdb,lagtime,n_clusters,CV_residues_index):
    if(os.path.isfile('data_output.pkl')):
        with open('data_output.pkl','rb') as in_data:
            data_output = pickle.load(in_data)
            
    feat = pyemma.coordinates.featurizer(pdb)
    all_pairs = feat.pairs(feat.select_Ca())
    ref = md.load_pdb(pdb)
    alpha = ref.top.select_atom_indices('alpha')
    data_index = []
    for cv_index in CV_residues_index:
        indices = ref.topology.select('resi '+' '.join(str(i) for i in cv_index))
        indices = np.intersect1d(alpha,indices)
        cv_pairs = feat.pairs(indices)
        # print('cv_pairs=',cv_pairs)
        index = np.array([np.where(np.sum(all_pairs==p,axis=1)==2)[0] for p in cv_pairs])
        data_index.append(index.reshape(-1,))
    
    tica_estimator = TICA(lagtime=lagtime, dim=2)
    cv_free_energies = []
    for index in data_index:
        # print('index=',index)
        selected_data = list(np.array(data_output)[:,:,index])
        tica = tica_estimator.fit(selected_data).fetch_model()
        # print('TICA dimension = ', tica.output_dimension)
        tics = tica.transform(selected_data)
        tmp_n_clusters = n_clusters
        while True:
            try:
                # print('tmp_n_clusters=',tmp_n_clusters)
                cluster = KMeans(tmp_n_clusters, max_iter=200).fit_fetch(np.concatenate(tics)[::10])
                dtrajs = [cluster.transform(traj) for traj in tics]
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
                tmp_n_clusters -= 1
                print("Error occurred:", str(e))
                print("Exception handling,tmp_n_clusters=",tmp_n_clusters)
        
        print('stat_dis=',stat_dis)
        d_awards = np.array([np.min(stat_dis[np.unique(dtraj)]) for dtraj in dtrajs])
        cv_free_energies.append(-np.log(d_awards))
    
    cv_free_energies = np.array(cv_free_energies)
    
    return cv_free_energies
    

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="Calculate scoresfrom structure & MSM model",
                                epilog="""
Example: run the full suite of benchmarks for the CUDA platform, printing the results as a table

    python benchmark.py --platform=CUDA --style=table

Example: run the apoa1pme benchmark for the CPU platform with a reduced cutoff distance

    python benchmark.py --platform=CPU --test=apoa1pme --pme-cutoff=0.8

Example: run the full suite in mixed precision mode, saving the results to a YAML file

    python benchmark.py --platform=CUDA --precision=mixed --outfile=benchmark.yaml""")

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


ad_methods = ['Multi_stage_TS','Adaptive_CV_TS','CVgenerator_TS']
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
    elif args.advanced_methods == 'CVgenerator_TS':
        from scipy.stats import skew
        native_ref = md.load_pdb(args.pdb)
        
        # parameter setting
        windows_size = [5,8,12,16,20,24,28,32]
        steps = [3,4,6,8,10,12,14,16]
        
        cache_scores = []
                
        cvgenerator_ts = CVgenerator(args.pdb,windows_size='auto',steps='auto',mode='regular')
        sel_indices, CV_residues_index = cvgenerator_ts.generateCV()
        
        
        # calculate structure weight
        num_hbonds = []
        pdb_files = os.listdir('./CV_structures/')
        for ifile in range(len(pdb_files)):
            ref = md.load_pdb('./CV_structures/CV_structure_'+str(ifile)+'.pdb')
            h_bond = md.baker_hubbard(ref,distance_cutoff=0.25)
            num_hbonds.append(h_bond.shape[0])
        num_hbonds = np.array(num_hbonds).astype(float)
        print('Number H-bonds in CVs : ',num_hbonds)
        # num_hbonds[num_hbonds==0] = 0.05
    
        # print('prior theta: ',prior_theta)
        if len(whole_traj)<(interval*args.n_replicas):
            sel_trajs = whole_traj
        else:
            sel_trajs = whole_traj[-interval*args.n_replicas:]
        traj_slices = make_slice_trajs(sel_trajs)
        print('len_traj_slices=',len(traj_slices))
        print("traj_slices=",traj_slices)
        
        cv_gen_rmsd = []
        for sel_indice in sel_indices:
            pool = multiprocessing.Pool(processes=args.n_replicas)
            rmsd_results = pool.map(cal_rmsd_score, [(traj_slice, native_ref, sel_indice) for traj_slice in traj_slices])
            pool.close()
            pool.join()
            rmsd_value = np.concatenate(rmsd_results)
            cv_gen_rmsd.append(rmsd_value)
        if len(whole_traj)<(interval*args.n_replicas):
            cache_scores += cv_gen_rmsd
            with open('CV_gen_rmsd.pkl','wb') as w:
                pickle.dump(cv_gen_rmsd,w)
        else:
            with open('CV_gen_rmsd.pkl','rb') as r:
                CV_gen_rmsd = pickle.load(r)
            for i in range(len(CV_gen_rmsd)):
                CV_gen_rmsd[i] = np.append(CV_gen_rmsd[i],cv_gen_rmsd[i])
            cache_scores += CV_gen_rmsd
            with open('CV_gen_rmsd.pkl','wb') as w:
                pickle.dump(CV_gen_rmsd,w)
        
        
        # # compute current degree of freedoms of each CVs
        pool = multiprocessing.Pool(processes=args.n_replicas)
        dof_results = pool.map(compute_current_freedoms, [(traj_slice, native_ref, CV_residues_index, 'SS') for traj_slice in traj_slices])
        pool.close()
        pool.join()
        dof_value = np.min(np.array(dof_results),axis=0).reshape(1,-1)
        # print('DOF=',dof_value)
        
        if os.path.isfile('CV_DOF.npy'):
            with open('CV_DOF.npy','rb') as r:
                cv_dof = np.load(r)
            cv_dof = np.concatenate([cv_dof,dof_value],axis=0)
        else:
            cv_dof = dof_value
        with open('CV_DOF.npy','wb') as w:
            np.save(w, cv_dof)
        CV_DOF = np.min(cv_dof,axis=0)
        print('CV_DOF=',CV_DOF)
        
        prefolding_flag = 1
        if os.path.isfile('prefolding_flag.npy'):
            with open('prefolding_flag.npy','rb') as r:
                prefolding_flag = np.load(r)
        if prefolding_flag:
            prefolding_flag = is_pre_folding_stage(sel_trajs, native_ref)
            with open('prefolding_flag.npy','wb') as w:
                np.save(w, prefolding_flag)
        
        # # compute unbiased prior weights
        # active_set = np.union1d(active_set,np.where(CV_DOF<2)[0])
        cv_scores_mins = np.array([np.min(score) for score in cache_scores])
        all_dof = np.array([len(i) for i in CV_residues_index])
        min_dof = np.min(all_dof)
        active_set = np.where(all_dof==min_dof)[0]
        active_set = np.intersect1d(active_set,np.where(cv_scores_mins>0.1)[0])
        active_bound = (0.08,0.2)
        active_step = 0.001
        active_gap = 0
        active_unique_res = []
        while len(active_unique_res)<native_ref.top.n_residues:
            active_set = np.append(active_set,np.intersect1d(np.where(cv_scores_mins<=active_bound[1]+active_gap)[0],np.where(cv_scores_mins>active_bound[0])[0]))
            active_set = np.unique(active_set)
            active_unique_res = np.array([])
            for i in active_set:
                active_unique_res = np.append(active_unique_res,CV_residues_index[i])
            active_unique_res = np.unique(active_unique_res)
            active_gap += active_step
        active_set = np.unique(active_set)
        
        print('Active CV bound = (0.08,',active_bound[1]+active_gap-active_step,')')
        print('Using ',len(active_set),' CVs')
        
        active_CV_residues_index = [CV_residues_index[i] for i in active_set]
        prior_theta = compute_prior_theta(active_CV_residues_index)
        prior_theta = prior_theta/np.sum(prior_theta)
        optimized_I = np.zeros(len(cache_scores))
        optimized_I[active_set] = prior_theta
        print("Optimized prior weights:", optimized_I)
        
        # # compute folding constrain factor
        pool = multiprocessing.Pool(processes=args.n_replicas)
        constrain_factor_results = pool.map(folding_constrain_factor, [(traj_slice, native_ref) for traj_slice in traj_slices])
        pool.close()
        pool.join()
        constrain_factor = np.concatenate(constrain_factor_results)
        if len(whole_traj)<(interval*args.n_replicas):
            Cf = constrain_factor
            with open('folding_constrain_factors.npy','wb') as w:
                np.save(w, Cf)
        else:
            with open('folding_constrain_factors.npy','rb') as r:
                Cf = np.load(r)
            Cf = np.append(Cf,constrain_factor)
            with open('folding_constrain_factors.npy','wb') as w:
                np.save(w, Cf)
        Cf = Cf**(-0.5)
        Cf = Cf/np.max(Cf)
        
        # # compute hydrophobic_score factor
        hydrophobic_res = hydrophobic_score_residue(args.pdb)
        print('hydrophobic residues:',hydrophobic_res)
        pool = multiprocessing.Pool(processes=args.n_replicas)
        hydrophobic_results = pool.map(hydrophobic_score_factor, [(args.pdb, traj_slice, hydrophobic_res, CV_residues_index) for traj_slice in traj_slices])
        pool.close()
        pool.join()
        hydrophobic_value = np.concatenate(hydrophobic_results,axis=1)
        if len(whole_traj)<(interval*args.n_replicas):
            hydrophobic_factors = hydrophobic_value
            with open('hydrophobic_factors.npy','wb') as w:
                np.save(w, hydrophobic_factors)
        else:
            with open('hydrophobic_factors.npy','rb') as r:
                hydrophobic_factors = np.load(r)
            hydrophobic_factors = np.concatenate((hydrophobic_factors,hydrophobic_value),axis=1)
            # hydrophobic_factors = np.append(hydrophobic_factors,hydrophobic_value)
            with open('hydrophobic_factors.npy','wb') as w:
                np.save(w, hydrophobic_factors)
        
        # # compute CV free energies
        # cv_free_energies = compute_cv_free_energy(args.pdb,args.lagtime,args.n_clusters,active_CV_residues_index)
        
        # # compute 3d-stacking factor
        global_sasa = md.shrake_rupley(native_ref)
        local_sasa = md.shrake_rupley(sel_trajs)
        print('local_sasa.shape=',local_sasa.shape)
        stacking_factor = np.sum(np.abs(local_sasa-global_sasa),axis=1)
        if len(whole_traj)<(interval*args.n_replicas):
            stacking_factors = stacking_factor
            with open('stacking_factors.npy','wb') as w:
                np.save(w, stacking_factors)
        else:
            with open('stacking_factors.npy','rb') as r:
                stacking_factors = np.load(r)
            stacking_factors = np.append(stacking_factors,stacking_factor)
            with open('stacking_factors.npy','wb') as w:
                np.save(w, stacking_factors)
        norm_stacking_factors = stacking_factors**(0.5)
        norm_stacking_factors = norm_stacking_factors/np.max(norm_stacking_factors)
        
        # # compute post-folding CVs
        post_folding_CV = is_post_folding_CV(native_ref, CV_residues_index)
        print("post_folding_CV",post_folding_CV)
        
        
        # # only filter in post-folding stage
        # if not prefolding_flag:
        #     filtered_index = filter_post_folding_index(sel_trajs, native_ref, args.n_clusters)
        #     filtered_index += len(whole_traj)-(args.n_replicas*interval)
        #     if os.path.isfile('filtered_post_folding_index.npy'):
        #         with open('filtered_post_folding_index.npy','rb') as r:
        #             post_folding_index = np.load(r)
        #         filtered_index = np.append(post_folding_index,filtered_index)
        #     with open('filtered_post_folding_index.npy','wb') as w:
        #         np.save(w, filtered_index)
        
        # filter in both stage
        filtered_index = filter_folding_index(sel_trajs, native_ref, args.n_clusters, simplified=True)
        
        CVgen = True
        reduce_uncertainty = False
        # agent = 'learn_alpha'
        agent = 'optimize_alpha'
        # agent = 'learn_W'
        # agent = 'predict'
        if CVgen:
            # CV_gen
            theta = []
            delt = []
            left_skewness = []
            regrat = []
            # max_score = 1.5*np.max(np.concatenate(cache_scores))
            # if len(his_actions)>0:
            #     activation_cutoff = np.array([len(cv)*0.2 for cv in CV_residues_index])
            #     his_pdbs = whole_traj[np.concatenate(his_actions)]
            #     his_cv_dof = compute_traj_frame_freedoms(traj=his_pdbs,ref=native_ref,CV_index=CV_residues_index)
            #     num_reached = np.sum((his_cv_dof-activation_cutoff)<=CV_DOF,axis=0)
            #     num_reached[num_reached==0]=1
            #     activation_portion = num_reached/((cv_dof.shape[0]-np.argmin(cv_dof,axis=0))*args.n_replicas)
            # else:
            #     activation_portion = np.ones(len(CV_residues_index))
            
            if not prefolding_flag:
                cache_scores = np.exp(hydrophobic_factors)*np.array(cache_scores)
            midfolding_flag = False
            postfolding_flag = False
            for n,score in enumerate(cache_scores):
                # # 1.
                # if np.min(score)==0:
                #     theta.append(-np.log(0.0001))
                # else:
                #     theta.append(-np.log(np.min(score)))
                
                # # 2.
                # start_values = score[::interval]
                # if start_values.shape[0]>0:
                #     reweighted_index = unbiased_distribution_index(start_values)
                #     discrete_scores = score.reshape(-1,interval)
                #     trajs_std = np.std(discrete_scores[reweighted_index],axis=1)
                # else:
                #     trajs_std = np.std(score)
                # theta.append(-np.log(np.max(score)/max_score)/np.mean(trajs_std))
                
                # # 3.
                # p = num_hbonds[n]/len(CV_residues_index[n])
                # q = np.min(score)/np.exp(0.4*len(CV_residues_index[n]))
                
                # # 4.
                # if n in active_set:
                #     disorder_DOF = CV_DOF[n]
                #     order_DOF = len(CV_residues_index[n])-disorder_DOF
                #     if prefolding_flag:
                #         p = num_hbonds[n]/len(CV_residues_index[n])
                #         # if np.min(score)<0.2:
                #         #     order_factor = np.exp(order_DOF)
                #         #     # order_factor = order_DOF
                #         # else:
                #         #     order_factor = np.exp(-disorder_DOF)
                #         order_factor = np.exp(order_DOF)
                #         q = np.min(score)*order_factor
                #         theta.append(p*q)
                #     else:
                #         # order_factor = np.exp(disorder_DOF)
                #         order_factor = np.exp(-len(CV_residues_index[n]))
                #         q = np.min(score*np.exp(hydrophobic_factors[n]))*order_factor
                #         theta.append(q)
                        
                #     if prefolding_flag & post_folding_CV[n]:
                #         theta[-1] = 0
                # else:
                #     theta.append(0)
                    
                # # 5.
                # if n in active_set:
                #     fe_index = np.argmin(score)//interval
                #     free_energy = cv_free_energies[np.where(active_set==n)[0],fe_index][0]
                #     theta.append(free_energy)
                # else:
                #     theta.append(0)
                
                # 6.
                arg_x = np.argmin(score)
                nth_traj = arg_x//interval
                seg_data = score[nth_traj*interval:(nth_traj+1)*interval]
                Relative_fluctuation = np.sqrt(np.mean(seg_data**2)-np.mean(seg_data)**2)/np.mean(seg_data)

                if n in active_set:
                    disorder_DOF = CV_DOF[n]
                    order_DOF = len(CV_residues_index[n])-disorder_DOF
                    if prefolding_flag:
                        order_factor = np.exp(-disorder_DOF)
                        p = num_hbonds[n]/len(CV_residues_index[n])
                        q = order_factor/Relative_fluctuation
                        theta.append(p*q)
                    elif (not prefolding_flag) and (np.max(np.min(hydrophobic_factors[np.min(hydrophobic_factors,axis=1)>0][:filtered_index],axis=1))>0.5):
                        midfolding_flag = True
                        order_factor = np.exp(-disorder_DOF)
                        q = order_factor/Relative_fluctuation
                        theta.append(q)
                    elif (not prefolding_flag) and (np.max(np.min(hydrophobic_factors[np.min(hydrophobic_factors,axis=1)>0][:filtered_index],axis=1))<=0.5):
                        postfolding_flag = True
                        order_factor = np.exp(-len(CV_residues_index[n]))
                        # p = np.exp(np.min(hydrophobic_factors[n]))
                        q = order_factor/Relative_fluctuation
                        theta.append(q)
                        
                    if prefolding_flag & post_folding_CV[n]:
                        theta[-1] = 0
                else:
                    theta.append(0)
                
                
                
                # # method1 : left skewness
                # left_skewness = left_skew(score)
                # delt.append(left_skewness)
                # if left_skewness<0.5:
                #     delt.append(0)
                # else:
                #     delt.append(left_skewness)
                
                # #  method2 : minimum raletive std
                # raletive_std = min_relative_std(x=score,interval=interval)
                # delt.append(raletive_std)
                
                # # method3 : optimal deviation
                # deviation = optimal_deviation(x=score,interval=interval,n_replicas=args.n_replicas)
                # delt.append(deviation)
                
                # # method4 : directional KL divergence
                # arg_score = np.argmin(score)
                # nth_traj = arg_score//interval
                # new_data = score[nth_traj*interval:(nth_traj+1)*interval]
                # KL_D = directional_relative_entropy(score,new_data)
                # delt.append(KL_D)
                
                # method5 : regrat
                epsilon = 0.003
                psai = 1
                # constrain_DOF = len(CV_residues_index[n])-CV_DOF[n]
                # psai = 1/(np.exp(constrain_DOF))
                if (os.path.isfile('weighted_factors.pkl')):
                    with open('weighted_factors.pkl', 'rb') as r:
                        weighted_factors = pickle.load(r)
                    N_try = 0
                    global_min_score = np.min(score)
                    arg_x = np.argmin(score)
                    nth_traj = arg_x//interval
                    seg_data = score[nth_traj*interval:(nth_traj+1)*interval]
                    # global_mean_min_score = np.mean(seg_data)
                    global_percentile_score = np.percentile(seg_data,90)
                    
                    discret_score = score.reshape(-1,interval)
                    initial_nseg = discret_score.shape[0]%args.n_replicas
                    discret_score = discret_score[initial_nseg::]
                    local_min_score = np.min(discret_score,axis=1)
                    round_ids1 = np.where((local_min_score-global_min_score)<epsilon)[0]
                    discret_seed = discret_score[:,0]
                    round_ids2 = np.where(discret_seed<global_percentile_score)[0]
                    round_ids = np.intersect1d(round_ids1,round_ids2)

                    D = 0
                    if round_ids.shape[0]>0:
                        D = np.sum([weighted_factors[id//args.n_replicas][n]*psai for id in round_ids])
                    
                    # if os.path.isfile('res_record.npy'):
                    #     with open('res_record.npy','rb') as r:
                    #         res_record = np.load(r)
                        

                    delt.append(-D)
                else:
                    weighted_factors = []
                    delt.append(0.5)
                
                # # method 6 : active
                # if (os.path.isfile('weighted_factors.pkl')):
                #     with open('weighted_factors.pkl', 'rb') as r:
                #         weighted_factors = pickle.load(r)
                # else:
                #     weighted_factors = []
                # delt.append(-np.log(activation_portion[n]))
                
            #     # method7 : 1+5
            #     if n in active_set:
            #         epsilon = 0.005
            #         psai = 1
            #         # constrain_DOF = len(CV_residues_index[n])-CV_DOF[n]
            #         # psai = 1/(np.exp(constrain_DOF))
            #         if (os.path.isfile('weighted_factors.pkl')):
            #             l_skew = left_skew(score)
            #             if l_skew<0.4:
            #                 left_skewness.append(0)
            #             else:
            #                 left_skewness.append(l_skew)
            #             with open('weighted_factors.pkl', 'rb') as r:
            #                 weighted_factors = pickle.load(r)
            #             N_try = 0
            #             global_min_score = np.min(score)
            #             arg_x = np.argmin(score)
            #             nth_traj = arg_x//interval
            #             seg_data = score[nth_traj*interval:(nth_traj+1)*interval]
            #             # global_mean_min_score = np.mean(seg_data)
            #             global_percentile_score = np.percentile(seg_data,90)
                        
            #             discret_score = score.reshape(-1,interval)
            #             initial_nseg = discret_score.shape[0]%args.n_replicas
            #             discret_score = discret_score[initial_nseg::]
            #             local_min_score = np.min(discret_score,axis=1)
            #             round_ids1 = np.where((local_min_score-global_min_score)<epsilon)[0]
            #             discret_seed = discret_score[:,0]
            #             round_ids2 = np.where(discret_seed<global_percentile_score)[0]
            #             round_ids = np.intersect1d(round_ids1,round_ids2)

            #             D = 0
            #             if round_ids.shape[0]>0:
            #                 D = np.sum([weighted_factors[id//args.n_replicas][n]*psai for id in round_ids])
                        
            #             # if os.path.isfile('res_record.npy'):
            #             #     with open('res_record.npy','rb') as r:
            #             #         res_record = np.load(r)
                            

            #             regrat.append(D)
            #         else:
            #             weighted_factors = []
            #             left_skewness.append(0.5)
            #             regrat.append(0.5)
            #     else:
            #         left_skewness.append(0)
            #         regrat.append(0)
                
            # left_skewness = np.array(left_skewness)/np.sum(left_skewness)
            # regrat = np.array(regrat)/np.sum(regrat)
            # delt = left_skewness-regrat
            
            # award_factors = []
            # punish_factors = []
            # if (os.path.isfile('award_factors.pkl')):
            #     with open('award_factors.pkl', 'rb') as r:
            #         award_factors = pickle.load(r)
            # if (os.path.isfile('punish_factors.pkl')):
            #     with open('punish_factors.pkl', 'rb') as r:
            #         punish_factors = pickle.load(r)
            # award_factors.append(left_skewness)
            # punish_factors.append(regrat)
            # with open('award_factors.pkl', 'wb') as w:
            #     pickle.dump(award_factors, w)
            # with open('punish_factors.pkl', 'wb') as w:
            #     pickle.dump(punish_factors, w)
            
                
            delt = np.array(delt).reshape(-1,)
            # print('theta=',theta)
            theta = optimized_I*np.array(theta)
            theta = theta/np.sum(theta)
            print('theta=',theta)
            print('delt=',delt)
            
            if agent == 'learn_alpha':
                # # method1 : based on skewness
                # if(os.path.isfile('agent_factor_loss.pkl')):
                #     with open('agent_factor_loss.pkl', 'rb') as r:
                #         agent_factor_loss = pickle.load(r)
                #     if len(agent_factor_loss[0])==len(agent_factor_loss[1])+1:
                #         loss = cal_loss(theta,cache_scores,args.n_replicas,interval,method="left_skewness")
                #         agent_factor_loss[1].append(loss)
                #     if np.sum(delt)==0:
                #         ALPHA = 1.0
                #     else:
                #         ALPHA = update_agent_factor(agent_factor_loss,method="left_skewness")
                #         agent_factor_loss[0].append(ALPHA)
                # else:
                #     agent_factor_loss = [[],[]]
                #     ALPHA = 1.0
                #     agent_factor_loss[0].append(ALPHA)
                # with open('agent_factor_loss.pkl', 'wb') as w:
                #     pickle.dump(agent_factor_loss,w)
                    
                # method2 : based on relative entropy distribution
                alpha = native_ref.top.select_atom_indices('alpha')
                sel_traj = whole_traj.atom_slice(alpha)
                cluster_labels = cluster(sel_traj.xyz,args.n_clusters)
                prior_ALPHA,loss = cal_loss(theta,cache_scores,args.n_replicas,interval,delt=delt,method="relative_entropy",cluster_labels=cluster_labels)
                agent_factor_loss = [prior_ALPHA,loss]
                ALPHA = update_agent_factor(agent_factor_loss,method="relative_entropy")
                print('Predicted ALPHA =',ALPHA)
                with open('agent_factor_loss.pkl', 'ab') as w:
                    pickle.dump(agent_factor_loss,w)
                
                # ALPHA = 0.3
                W = []
                for n,i in enumerate(theta):
                    # W.append(i*np.exp(ALPHA*delt[n])) #  method1 : left skewness
                    # W.append(-i*np.log2(ALPHA*delt[n])) #  method2 : minimum raletive std
                    W.append(i*np.exp(ALPHA*delt[n])) # method3 : optimal deviation
                W = np.array(W)/np.sum(W)
                W = W.reshape(1,-1)
                print('W=',W)
                
                # weighted_factors.append(W.reshape(-1,))
                # with open('weighted_factors.pkl', 'wb') as w:
                #     pickle.dump(weighted_factors, w)
                CVgen_score_pre = np.dot(W, np.array(cache_scores)).reshape(-1,)
                CVgen_score = CVgen_score_pre
                CV_indices = np.arange(cache_scores[0].shape[0])
                
                # # filter CV_score
                # father_nodes, CV_topology = find_CV_topology(n_patch,CV_residues_index)
                # D_cutoff = [compute_decision_cutoff(i) for i in cache_scores]
                # CV_indices = filter_CVscore_with_correlation(cache_scores,W,CVgen_score_pre,D_cutoff,args.n_clusters,method='prior',candidate_score_scut_off=0.3,father_nodes=father_nodes,CV_topology=CV_topology)
                # alpha = native_ref.top.select_atom_indices('alpha')
                # sel_traj = whole_traj[CV_indices].atom_slice(alpha)
                # sel_CV_score = CVgen_score_pre[CV_indices]
                
                # if reduce_uncertainty and len(his_actions)>0:
                #     # calculate latent score for all CV_scores
                #     actions_in_reduced_region = np.intersect1d(CV_indices,np.concatenate(his_actions))
                #     if actions_in_reduced_region.shape[0]>0:
                #         mapped_actions = np.array([np.where(CV_indices==action)[0] for action in actions_in_reduced_region]).reshape(-1,)
                #         cluster_labels = cluster(sel_traj.xyz,args.n_clusters)
                #         latent_scores = cal_latent(cluster_labels,sel_CV_score,mapped_actions)
                #         sel_CV_score = latent_scores
                # CVgen_score = sel_CV_score 
                print('CVgen_score (min,mean,max):',np.min(CVgen_score_pre),np.mean(CVgen_score_pre),np.max(CVgen_score_pre))
                   
            elif agent == 'optimize_alpha':
                ALPHA = optimize_ALPHA(CV_residues_index,theta,cache_scores,args.n_replicas,interval,delt)
                # if np.max(np.abs(delt))==0:
                #     ALPHA = 0.5
                # else:
                #     ALPHA = 3/np.abs(np.dot(theta.reshape(1,-1),np.abs(delt).reshape(-1,1)))
                #     # ALPHA = 3/np.max(np.abs(delt))
                print('Optimized ALPHA =',ALPHA)
                W = []
                for n,i in enumerate(theta):
                    # W.append(i*np.exp(ALPHA*delt[n])) #  method1 : left skewness
                    # W.append(-i*np.log2(ALPHA*delt[n])) #  method2 : minimum raletive std
                    W.append(i*np.exp(ALPHA*delt[n])) # method3 : optimal deviation
                W = np.array(W)/np.sum(W)
                W = W.reshape(1,-1)
                print('W=',W)
                
                weighted_factors.append(W.reshape(-1,))
                with open('weighted_factors.pkl', 'wb') as w:
                    pickle.dump(weighted_factors, w)
                
                if prefolding_flag:
                    CVgen_score_pre = np.dot(W, np.array(cache_scores)).reshape(-1,)
                    # stag 1 : pre-folding
                    print('Pre-folding')
                    CVgen_score = Cf*CVgen_score_pre
                    # CVgen_score = CVgen_score_pre
                    CV_indices = np.arange(CVgen_score.shape[0])
                    # CVgen_score = CVgen_score[filtered_index]
                    # CV_indices = filtered_index
                else:
                    CVgen_score_pre = np.dot(W, np.array(cache_scores)).reshape(-1,)
                    
                    # stag 2 : 3d-stacking folding
                    if minfolding_flag:
                        print('Mid-folding')
                    elif postfolding_flag:
                        print('Post-folding')
                    # CVgen_score = norm_stacking_factors*CVgen_score_pre
                    # CVgen_score = np.exp(2*hydrophobic_factors)*CVgen_score_pre
                    CVgen_score = CVgen_score_pre
                    CVgen_score = CVgen_score[filtered_index]
                    CV_indices = filtered_index
                
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
            
            if reduce_uncertainty and len(his_actions)>0:
                # calculate latent score for all CV_scores
                actions_in_reduced_region = np.intersect1d(CV_indices,np.concatenate(his_actions))
                if actions_in_reduced_region.shape[0]>0:
                    mapped_actions = np.array([np.where(CV_indices==action)[0] for action in actions_in_reduced_region]).reshape(-1,)
                    heavy = native_ref.top.select_atom_indices('heavy')
                    sel_traj = whole_traj[CV_indices].atom_slice(heavy)
                    cluster_labels = cluster(sel_traj.xyz,args.n_clusters)
                    latent_scores = cal_latent(cluster_labels,CVgen_score,mapped_actions)
                    sel_CV_score = latent_scores
                CVgen_score = sel_CV_score 
                
            sel_score = CVgen_score.reshape(-1,)
            sel_index = CV_indices
            arms_values, arms_indices = thompson_sampling_MSM(args.lagtime,args.n_clusters,prior_score=sel_score,prior_score_indices=sel_index)
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
        
        rmsd = md.rmsd(whole_traj, native_ref, frame=0, atom_indices=alpha)
        whole_traj[np.argmin(rmsd)].save_pdb('min_rmsd.pdb')
        print("Minimum rmsd is",np.min(rmsd))
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
                
elif args.advanced_methods == 'TS_GP':
    if(os.path.isfile('his_actions.pkl')):
        his_actions_con = np.concatenate(his_actions)
    native_ref = md.load_pdb(args.pdb)
    heavy = native_ref.top.select_atom_indices('heavy')
    alpha = native_ref.top.select_atom_indices('alpha')
    interval = args.sampling_interval
    max_score=False
    
    tics = get_tica(1,args.lagtime)
    tica = np.concatenate(tics)

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
            
    if args.prior == 'initial':
        arr = np.arange(len(whole_traj))[::interval]
        posterior = []
        for CV_score in cache_scores:
            posterior.append(get_posterior(arr, CV_score, interval, max_score=max_score))
        posterior = np.array(posterior)
        CV_id = 0
        dynamic_prior = get_dynamic_prior(cache_scores[CV_id],tica,interval,max_score=max_score,scale_dynamic_prior='linear')
        with open('posterior.npy', 'wb') as w:
            np.save(w, posterior)

        if args.TS_GP_mode == 'multi':
            old_paths = [[],[],[],[],[]]
            old_paths[0].append(arr)                # --  a list of 'x' value
            old_paths[1].append(arr//interval)      # --  a list of 'y' index
            old_paths[2].append(posterior[0])          # --  a list of 'y' value 
            old_paths[3].append(arr//interval+1)      # --  a list of 'y' index in real path
            old_paths[4].append(np.array([interval]*arr.shape[0]))                # --  a list of 'x' index in real path
            with open('paths.pkl', 'wb') as w:
                pickle.dump(old_paths, w)
            
    elif args.prior == 'update':
        with open('posterior.npy', 'rb') as r:
            prior = np.load(r)
        arr = np.arange(len(whole_traj))[::interval]
        arr = arr[-args.n_replicas:]
        posterior = []
        delt = []
        for CV_id,CV_score in enumerate(cache_scores):
            know = get_posterior(arr, CV_score, interval, max_score=max_score)
            posterior.append(np.append(prior[CV_id], know))
            delt.append(np.abs(np.min(posterior[CV_id])-np.min(prior[CV_id])))
        posterior = np.array(posterior)
        if np.max(delt)==0:
            delt = [random.randint(1, 100) for _ in range(len(cache_scores))]
        delt = np.array(delt).reshape(-1,)
        CV_id = np.argmax(delt)
        print('*********** Select ',matrixs_name[CV_id],'as metrix ***********')
        
        dynamic_prior = get_dynamic_prior(cache_scores[CV_id],tica,interval,max_score=max_score,scale_dynamic_prior='linear')
        with open('posterior.npy', 'wb') as w:
            np.save(w, posterior)
        if args.TS_GP_mode == 'multi':
            with open('paths.pkl', 'rb') as r:
                old_paths = pickle.load(r)

            his_id = [i//interval for i in his_actions_con[-args.n_replicas:]]
            his_frames = [i%interval for i in his_actions_con[-args.n_replicas:]]
            new_connect_top = [np.array(his_id), np.array(his_frames)]
            for i in range(args.n_replicas):
                connect_top = [new_connect_top[0][-args.n_replicas+i],new_connect_top[1][-args.n_replicas+i],posterior[0].shape[0]-args.n_replicas+i]
                old_paths = update_multi_paths(old_paths=old_paths,new_connect_top=connect_top,interval=interval,posterior=posterior[0])
            with open('paths.pkl', 'wb') as w:
                pickle.dump(old_paths, w)
        
    unit_len = interval * args.n_replicas
    
    if args.TS_GP_mode == 'multi':
        old_paths_0 = make_slice(old_paths[0])
        old_paths_2 = make_slice(old_paths[2])
        old_paths_3 = make_slice(old_paths[3])
        old_paths_4 = make_slice(old_paths[4])
        
        if len(old_paths[0])<args.n_replicas:
            pool = multiprocessing.Pool(processes=len(old_paths[0]))
            results = pool.map(multi_sample_GP_TS, [(paths[0], paths[1], paths[2], paths[3], unit_len, args.n_replicas, 100000, interval) for paths in zip(old_paths_0,old_paths_2,old_paths_3,old_paths_4)])
            pool.close()
            pool.join()
        else:
            pool = multiprocessing.Pool(processes=num_cores)
            results = pool.map(multi_sample_GP_TS, [(paths[0], paths[1], paths[2], paths[3], unit_len, args.n_replicas, 100, interval) for paths in zip(old_paths_0,old_paths_2,old_paths_3,old_paths_4)])
            pool.close()
            pool.join()
        
        sampled_X = [i[0] for i in results]
        sampled_Y = [i[1] for i in results]
        sampled_X = np.concatenate(sampled_X).astype(int)
        sampled_Y = np.concatenate(sampled_Y)
        
        # print(sampled_X)
        # print(sampled_Y)
        
        if max_score:
            idx = np.argsort(sampled_Y)[-args.n_replicas:]
        else:
            idx = np.argsort(sampled_Y)[:args.n_replicas]
        samples_index = sampled_X[idx]
    elif args.TS_GP_mode == 'single':
        if args.nth_round<1:
            samples_index = single_sample_GP_TS(tica, interval, posterior[CV_id], dynamic_prior, args.n_replicas)
        else:
            samples_index = single_sample_GP_TS(tica, interval, posterior[CV_id], dynamic_prior, args.n_replicas, process_data='denoise')
    
    # Avoid selecting nodes
    samples_index[np.where(samples_index%interval==0)] += 1
    # Avoid selecting his_actions
    if(os.path.isfile('his_actions.pkl')):
        cache_indices = np.zeros(interval, dtype=bool)
        his_actions_con_set = set(his_actions_con)
        for n, id in enumerate(samples_index):
            if id not in his_actions_con_set:
                continue
            st = (id // interval) * interval
            ed = st + interval
            arr1 = his_actions_con[his_actions_con > st]
            arr2 = his_actions_con[his_actions_con < ed]
            arr = np.intersect1d(arr1, arr2)
            arr -= st
            cache_indices[arr] = True
            cache_indices[id - st] = True
            argid = np.where(cache_indices == 0)[0]
            nextid = np.argmin(np.abs(id - st - argid))
            samples_index[n] = argid[nextid] + st
            his_actions_con = np.append(his_actions_con, samples_index[n])
        
    his_actions.append(samples_index)
    
    rmsd = md.rmsd(whole_traj, native_ref, frame=0, atom_indices=alpha)
    print("Minum rmsd is",np.min(rmsd))
    os.mkdir(args.outpdb_dir + '/round' + str(args.nth_round + 1) + '/')
    sampled_states = whole_traj[samples_index]
    for i, state in enumerate(sampled_states):
        state.save_pdb(args.outpdb_dir + '/round' + str(args.nth_round + 1) + '/' + str(i) + '.pdb')
        node = samples_index[i] // interval
        frame = samples_index[i] % interval
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
