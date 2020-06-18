"""
ecam Jul19

Test barrier with real data
"""

import numpy as np
import replication as re
import pickle as pk

default = {'C_at': 15.171875, 'N_origins': 446.0, 'e_dot': 0.09854296875, 'e_max': 0.9421875000000002, 'nbar': 0.0, 'nu': 1.0}

def get_origins(p):
    """
    p = [px,tt,300 (binsize)]
    """
    px = p[0]
    tt = p[1]
    binsize = int(p[2])
    N = int(p[3])

    loc = np.random.choice(np.arange(len(px)),N,True,px)

    #Remove repeated location
    loc = list(set(loc))

    #Get times
    times = np.random.choice(np.arange(len(tt)),len(loc),True,tt)

    #Re-scale
    #loc = [ int(x)*25/binsize  for x in loc ]


    #Origins.
    d = {}
    for x,t in zip(loc,times):
        xx = int(x)*25/binsize
        if xx not in d:
            d[xx] = t
        else:
            d[xx] = min( [ t, d[xx] ]) #if repeated, keep only first firing one

    #Generate list
    origins = [ [x,d[x]] for x in d ]

    #Sort and return
    return re.sort_origins_by_time(origins) 

def lt_barrier(o,x,p):
    """
    p[0] barrier delay
    """
    #Before the barrier
    if x > 11087:
        return o[1] + (o[0] - x)

    #Origin after
    if o[0] < 11087:
        return o[1] + (o[0] - x)
        
    #We are crossing the barrier 
    return p[0] + o[1] + (o[0] - x)

def rt_barrier(o,x,p):
    """
    p[1] barrier delay
    """

    #We are left (before) the barrier, or origin right of the barrier
    if x < 11087:
        return o[1] + (x - o[0])

    #Origin right of the barrier (after)
    if o[0] > 11087:
        return o[1] + (x - o[0])

    #We have to cross the barrier
    #Origin to barrier + barrier time + slowed barrier to x
    return o[1] + p[1] + (x - o[0])

def get_px(C_at,at_richness,transcript):
    p = [ t*np.exp(C_at*at) for t,at in zip(transcript,at_richness) ]

    p = np.array(p)
    #Manual suppression of origins
    p[ 3324*1000/25 : 3325*1000/25 ] = 0.
    p[ 3326*1000/25 : 3327*1000/25 ] = 0.
    p[ 3315*1000/25 : 3317*1000/25 ] = 0.
    p[ 3330*1000/25 : 3332*1000/25 ] = 0.

    #p[ 3330*1000/25 : 3335*1000/25 ] = 0.
    #p[ 3312*1000/25 : 3314*1000/25 ] = 0.
    #p[ int(3320.7*1000)/25 : 3322*1000/25 ] = 0.
    #p[ 3338*1000/25 : 3339*1000/25 ] = 0.
    #p[ 3301*1000/25 : 3303*1000/25 ] = 0.

    p = np.array(p)/sum(p)

    return p

def p_t(e_dot,e_max,t):
    t_cut = e_max / e_dot
    if t <= t_cut:
        return e_dot*t

    return e_max

def get_tt(e_max,e_dot):
    prob = [0]
    fails = 1.
    s = 0.
    t = 1

    while s < 1. - 1.e-3:
        p = p_t(e_dot,e_max,t)
        fp = fails*p
        prob.append(fp)
        fails = fails*(1-p)
        t = t+1
        s = s+fp

    return np.array(prob)/s 

def solve_barrier(chunk_processor,origin_param,speed_param,Nsamples):
    at_richness = np.load("chr1_at25.npy")
    transcript = np.load("chr1_trans25.npy")
    px = get_px(origin_param["C_at"],at_richness,transcript)
    tt = get_tt(origin_param["e_max"],origin_param["e_dot"])
    binsize = 300

    return re.do_experiment_serial(len(px)*25/binsize,get_origins,[px,tt,binsize,origin_param["N_origins"]],lt_barrier,rt_barrier,speed_param,chunk_processor,Nsamples)
def solve_dir_barrier(origin_param=default,speed_param=[0,0],Nsamples=100):
    return solve_barrier(re.get_direction,origin_param,speed_param,Nsamples)

   
 
