#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
from dataclasses import dataclass
from collections import deque
from sklearn.neural_network import MLPRegressor
import random
from itertools import product

def seqDist(s1, s2):
    return sum([1 if b1 != b2 else 0 for b1, b2 in zip(s1, s2)])

def mutateSeq(S, mut):
    pos, mut_base = mut           
    
    if pos >= len(S):
        raise Exception("Position out of range") 
    
    if pos == 0:
        return mut_base + S[1:]
    elif pos == len(S)-1:
        return S[:len(S)-1] + mut_base
    else:
        return S[:pos] + mut_base + S[pos+1:]

def makeMutations(S, mut):
    Sx1 = S
    for i in range(0, len(mut[0])):
        Sx2 = mutateSeq(Sx1, (mut[0][i], mut[1][1][i]))
        Sx1 = Sx2
    return Sx1

def hot1Encode(S):
    encoded = []
    dct_bases = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    for b in S:
        encoded.extend(dct_bases[b])
    return encoded
    
def assemble_hpyerparam_dicts(H):
    var_hp = sorted(H.keys())
    grid = list(product(*[H[x] for x in var_hp]))
    
    hpcs = []
    for i in range(0, len(grid)):
        dct_hp = {}
        for j in range(0, len(var_hp)):
            dct_hp[var_hp[j]] = grid[i][j]
        hpcs.append(dct_hp)
    return [(i, hpcs[i]) for i in range(0, len(hpcs))]

@dataclass
class BfsNode:
    seq: str
    cpm: int
    depth: int
    muts_from_start: int

@dataclass
class Mutant:
    seq: str
    mut: list
    pred_cpm: float

def mutantToNode(M, N):
    M_depth = N.depth+1
    M_muts_from_start = N.muts_from_start+len(M.mut[0])    
    return BfsNode(seq=M.seq, cpm=M.pred_cpm, depth=M_depth, muts_from_start=M_muts_from_start)

def beamSearch(start_seq_seq, start_seq_cpm, mutations, model, beam_width, mode, max_depth, exploration_cpm_threshold, top_explored):

    queue = deque([BfsNode(seq=start_seq_seq, cpm=start_seq_cpm, depth=0, muts_from_start=0)])
    explored_sequences = []
    
    while len(queue):
        
        # Get next node
        next_node = queue.popleft()
        
        # Make every possible mutation -- get mutant batch
        mutants_batch = [Mutant(seq=makeMutations(next_node.seq, mut), mut=mut, pred_cpm=None) for mut in mutations if ''.join([next_node.seq[x] for x in mut[0]]) == mut[1][0]]

        # Run mutants through the blacklist filter
        mutants_batch = [m for m in mutants_batch if m.seq not in blacklist]
        if len(mutants_batch) == 0:
            continue

        # Encode mutants
        mutants_encoded = np.array([np.array(hot1Encode(m.seq)) for m in mutants_batch])

        # Make predictions about mutants
        if len(mutants_encoded) == 1:
            mutants_encoded = mutants_encoded.reshape(1, -1)

        if mode == 'directed':
            predictions = model.predict(mutants_encoded)
        elif mode == 'random':
            predictions = np.random.rand(len(mutants_encoded))

        # Assign the predicted cpms
        for i in range(0, len(predictions)):
            mutants_batch[i].pred_cpm = predictions[i]

        # Convert mutants from mutant class to bfs node class and sort by cpm
        mutants_batch = [mutantToNode(m, next_node) for m in mutants_batch]
        mutants_batch = sorted(mutants_batch, key=lambda p: p.cpm, reverse=True)

        for i, m in enumerate(mutants_batch):
            if len(explored_sequences) < top_explored:
                    explored_sequences.append(m)
                
            else:
                if m.cpm > explored_sequences[-1].cpm:
                    del explored_sequences[-1]
                    explored_sequences.append(m)
                    explored_sequences = sorted(explored_sequences, key=lambda p: p.cpm, reverse=True)                        

            if (i < beam_width) and (m.depth <= max_depth):
                queue.append(m)

            blacklist.add(m.seq)
        
    return explored_sequences

beam_params = {
    'topN_start' : [1, 3, 10, 30],
    'beam_width' : [1, 2, 3, 4, 5],
    'max_depth' : [1, 2, 3, 4, 5, 6, 7],
    'top_explored' : [100], 
    'mode' : ['directed']
}

beam_params_grid = assemble_hpyerparam_dicts(beam_params)

df_res_metrics = pd.DataFrame(columns=['topN_start', 'beam_width', 'max_depth', 'top_explored', 'precision', 'recall', 'search_space'])
df_res_seqs = []

# Load the entire dataset
df = pd.read_csv('/home2/kurfurst/projects/jk/datasets/strc_km.csv', usecols=['varseq', 'cpm'])

# Make helpers
seq_2_cpm = {s : c for s, c in zip(df['varseq'], df['cpm'])}

# Load the splits
with open('/home2/kurfurst/projects/jk/ml/s7/testing_splits/splits.pkl', mode='rb') as sf:
    splits = pickle.load(sf)

# Load the possible mutations
with open('/home2/kurfurst/projects/jk/ml/s7/prep/allowed_mutations_permuations.pkl', mode='rb') as mf:
    mutations = pickle.load(mf)

# Load the trained model
with open('/home2/kurfurst/projects/jk/ml/s7/training_final/model_trained_final_trn2.pkl', mode='rb') as mf:
    model = pickle.load(mf)

sampling_i, sampling_value = 2, 0.01

# Get the sampled df
# [1, 0.1, 0.01, 0.001] These are the sampling of the training data
df_tst, df_val, df_trn = df.loc[splits['tst']], df.loc[splits['val']], df.loc[splits['trn'][sampling_i]]

# I'll be using the best training sequences as starting points so sorting the training dataframe
df_trn = df_trn.sort_values('cpm', ascending=False)

for q in range(0, len(beam_params_grid)):
    print(q)
    
    beam_search_params = beam_params_grid[q][1]

    # Run beamsearch
    blacklist = set(df_trn['varseq'].tolist())
    res_all = []
    for i in range(0, beam_search_params['topN_start']):
        res_start = beamSearch(df_trn.iloc[i]['varseq'], df_trn.iloc[i]['cpm'],
                               mutations=mutations, 
                               model=model,
                               mode=beam_search_params['mode'],
                               beam_width=beam_search_params['beam_width'],
                               max_depth=beam_search_params['max_depth'],
                               top_explored=beam_search_params['top_explored'],
                               exploration_cpm_threshold=df_trn.iloc[0]['cpm'])
        
        res_all.extend(res_start)
    
    df_res_all = pd.DataFrame([[r.seq, r.cpm, r.depth, r.muts_from_start] for r in res_all], columns=['varseq', 'predicted_cpm', 'depth', 'muts'])
    df_res_all = df_res_all.head(100)
    
    df_res_seqs.append(df_res_all)
    
    seqs_exp = set(df_res_all['varseq'].tolist())
    seqs_tst = set(df_tst['varseq'].tolist())
    
    precision = len(seqs_exp.intersection(seqs_tst)) / len(seqs_exp)
    #print('precision:', precision)
    
    recall = len(seqs_exp.intersection(seqs_tst)) / len(seqs_tst)
    #print('recall:', recall)
    
    search_space_size = len(blacklist) - len(df_trn)
    #print('search space size:', search_space_size)
    
    df_res_metrics.loc[len(df_res_metrics)] = [beam_search_params['topN_start'],
                                               beam_search_params['beam_width'],
                                               beam_search_params['max_depth'],
                                               beam_search_params['top_explored'],
                                               precision,
                                               recall,
                                               search_space_size
                                              ]
    df_res_seqs.append(df_res_all)

df_res_metrics.to_csv('beam_search_params_effect_SI_results.csv')
