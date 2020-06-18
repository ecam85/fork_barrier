"""
ecam May20

Test barrier with real data

Karel's barrier data.
"""

import numpy as np
import replication as re
import pickle as pk
import random

default = {'C_at': 15.171875, 'N_origins': 363.0, 'e_dot': 0.09854296875, 'e_max': 0.9421875000000002, 'nbar': 0.0, 'nu': 1.0}

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

def simloc(x,binsize=300):
    """
    Simulation units (index) location, from genomic coordinate in bp.
    """
    return int(x)/300 #Integer division

def lt_time_rts1(o,x,p):
    """
    Accounts for rDNA barrier 865150 ->866093
    p[0] = binsize
    p[1] = total delay for left travelling forks at barrier.
    """
    x0 = 865150
    x1 = 866093
    #Location before (to the right of the!) barrier
    if x > simloc(x1,p[0]):
        return o[1] + (o[0] - x) #Canonical speed
        

    #Origin after (to the left of!) the barrier:
    if o[0] < simloc(x0,p[0]):
        return o[1] + (o[0] - x) #Canonical speed

    #We are crossing the barrier.
    #Fraction of the barrier that we cross.
    f = min( [1., float(simloc(x1,p[0]) - x)/simloc(x1-x0,p[0]) ] )
    return f*p[1] + o[1] + (o[0] - x)

def rt_time_rts1(o,x,p):
    """
    p[0] = binsize
    p[1] = lt forks delay (used in rdna barrier, not here!)
    p[2] = rt forks delay
    p[4] = barrier efficiency (Fraction of stopped forks)
    """
    x0 = 851806
    x1 = 852664

    if len(p) >= 5:
        ignore_frac = 1 - p[4]
    else:
        ignore_frac = .25

    #1/4 of forks ignore the barrier
    if np.random.random() < ignore_frac:
        return o[1] + (x - o[0])

    #Before barrier (left of)
    if x < simloc(x0,p[0]):
        return o[1] + (x - o[0])

    #Origin after barrier (right of)
    if o[0] > simloc(x1,p[0]):
        return o[1] + (x - o[0])
    
    #We are crossing the barrier
    #Fraction of the barrier that we cross:
    f = min( [ 1., float(x - simloc(x0,p[0])) / simloc(x1-x0,p[0]) ] )
    return f*p[2] + o[1] + (x - o[0])

def rt_time_rts1rts1(o,x,p):
    """
    double rts1 barrier
    p[0] = binsize
    p[1] = lt forks delay (used in rdna barrier, not here!)
    p[2] = rt forks delay
    p[3] = restarted fork delay
    """
    x0 = 851806 
    x1 = 852664  
    y0 = 854425  
    y1 = 855283 

    #Before barrier (left of)
    if x < simloc(x0,p[0]):
        return o[1] + (x - o[0])

    #Origin after barrier (right of)
    if o[0] > simloc(y1,p[0]):
        return o[1] + (x - o[0])
    
    #We are crossing the barrier.
    restarted = True #The fork stop and restarted at the barrier.
    #Barrier 1:
    f1 = min( [ 1., float(x - simloc(x0,p[0])) / simloc(x1-x0,p[0]) ] )

    if np.random.random() < .25:
        f1 = 0. # 1/4 of forks ignore first barrier.
        restarted = False
    
    #Barrier2:
    if x < simloc(y0,p[0]):
        f2 = 0. #We have not reached barrier 2
    else:
        f2 = min( [ 1., float(x - simloc(y0,p[0])) / simloc(y1-y0,p[0]) ] )

    if restarted:
        t2 = p[3] #delay for restarted forks
    elif np.random.random() < .25:
        t2 = 0. #1/4 of forks ignore barrier
    else:
        t2 = p[2]

    return f1*p[2] + f2*t2 + o[1] + (x - o[0])

def get_polusage_rts1(o,x,lt_time,rt_time,params,chunk):
    """
    Returns the pol usage in watson and crick strands
    (watson is done by epsilon in canonical right moving forks)

    Exploits np.arrays!

    Args:
        o origin pair (location,time)
        x location to look at
        lt,rt,params as before
            params[0] = binsize
            params[2] = one barrier delay

    Returns:
        A len 2 np array with
            [1,1] both strands by epsilon
            [-1,-1] both by delta
            [1,-1] watson epsilon, crick delta
            [-1,1] watson delta, crick epsilon
            [0,0] - origin or on node termination
    """
    x0 = 851806
    x1 = 852664

    #x is an origin
    if o[0] == x:
        return np.array([0,0])

    #Check for on node termination.  
        if chunk[0][0] == x or chunk[2][0] == x:
            return np.array([0,0])

    #Right moving fork
    if o[0] < x:
        #The barrier is off, no fork reversal!
        if params[2] == 0:
            return np.array([1,-1])

        #We are left (before) the barrier, or origin right of the barrier
        if x < simloc(x0,params[0]) or o[0] > simloc(x1,params[0]):
            return np.array([1,-1]) #Canonical right moving fork
       #We cross the barrier. The fork restarted as delta-delta
        if chunk[2][1] > chunk[1][1] + (chunk[2][0] - chunk[1][0]): #Stop at the barrier for a while
            return np.array([-1,-1]) #Stop at the barrier -> restarted fork -> delta-delta
  
     
        #We cross the barrier but did not stop -> canonical fork. 
        return np.array([1,-1])


    #Left moving forks, they are just canonical.
    return np.array([-1,1])

def get_polusage_2xrts1(o,x,lt_time,rt_time,params,chunk):
    """
    Returns the pol usage in watson and crick strands
    (watson is done by epsilon in canonical right moving forks)

    Exploits np.arrays!

    Args:
        o origin pair (location,time)
        x location to look at
        lt,rt,params as before
            params[0] = binsize
            params[2] = one barrier delay

    Returns:
        A len 2 np array with
            [1,1] both strands by epsilon
            [-1,-1] both by delta
            [1,-1] watson epsilon, crick delta
            [-1,1] watson delta, crick epsilon
            [0,0] - origin or on node termination
    """
    x0 = 851806
    x1 = 852664
    y0 = 854425
    y1 = 855283

    #x is an origin
    if o[0] == x:
        return np.array([0,0])

    #Check for on node termination. #This can be improved: delta-delta fork vs delta-eps fork is [-1,0]
    if len(chunk) > 0:
        if chunk[0][0] == x or chunk[2][0] == x:
            return np.array([0,0])

    #Right moving fork
    if o[0] < x:
        #The barrier is off, no fork reversal!
        if params[2] == 0:
            return np.array([1,-1])
        
        #We are left (before) the first barrier, or origin right of the barrier second barrier
        if x < simloc(x0,params[0]) or o[0] > simloc(y1,params[0]):
            return np.array([1,-1]) #Canonical right moving fork
         #We cross the barrier. The fork restarted as delta-delta
        if chunk[2][1] > chunk[1][1] + (chunk[2][0] - chunk[1][0]): #Stop at the barrier for a while
            #We do not know at what barrier we stopped. Make it random:
            if np.random.random() < .75: #Stop at first barrier, delta delta all the way
                return np.array([-1,-1]) #Stop at the barrier -> restarted fork -> delta-delta
            elif x < simloc(y0,params[0]): #Have not reach 2nd barrier
                return np.array([1,-1])
            elif np.random.random() < .75: #stop at second barrier 
                return np.array([-1,-1]) #Stop at the barrier -> restarted fork -> delta-delta
 
     
        #We cross the barrier but did not stop -> canonical fork. 
        return np.array([1,-1])


    #Left moving forks, they are just canonical.
    return np.array([-1,1])

def rt_barrier_slow(o,x,p):
    """
    p[1] barrier delay
    p[2] speed reduction. Since canonical speed is 1, p[2] is also the new speed.
    """

    #We are left (before) the barrier, or origin right of the barrier
    if x < 2840:
        return o[1] + (x - o[0])

    #Origin right of the barrier (after)
    if o[0] > 2840:
        return o[1] + (x - o[0])

    #We have to cross the barrier
    #Origin to barrier + barrier time + slowed barrier to x
    return o[1] + (2840 - o[0]) + p[1] + (x - 2840)/p[2]

def get_px(C_at,at_richness,transcript):
    p = [ t*np.exp(C_at*at) for t,at in zip(transcript,at_richness) ]

    return np.array(p)/sum(p)

def get_px_supp(C_at,at_richness,transcript,suppress):
    """
    As before, but with supression of minor origins
    """
    p = np.array([ t*np.exp(C_at*at) for t,at in zip(transcript,at_richness) ])

    for x in suppress:
        p[int(x/25)] = 0

    return np.array(p)/sum(p)

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

def solve_barrier(chunk_processor,origin_param,speed_param,Nsamples,lt_barrier=lt_barrier,rt_barrier=rt_barrier,suppress=[0]):
    at_richness = np.load("chr2_at25.npy")
    transcript = np.load("chr2_trans25.npy")
    #px = get_px(origin_param["C_at"],at_richness,transcript)
    px = get_px_supp(origin_param["C_at"],at_richness,transcript,suppress)
    tt = get_tt(origin_param["e_max"],origin_param["e_dot"])
    binsize = speed_param[0]

    return re.do_experiment_serial(len(px)*25/binsize,get_origins,[px,tt,binsize,origin_param["N_origins"]],lt_barrier,rt_barrier,speed_param,chunk_processor,Nsamples)

def solve_pol_barrier_rts1(origin_param=default,speed_param=[300,0,0],Nsamples=100,suppress=[0]):
    return get_strand_pu(solve_barrier(get_polusage_rts1,origin_param,speed_param,Nsamples,lt_barrier=lt_time_rts1,rt_barrier=rt_time_rts1,suppress=suppress))

def solve_pol_barrier_2xrts1(origin_param=default,speed_param=[300,0,0],Nsamples=100,suppress=[0]):
    return get_strand_pu(solve_barrier(get_polusage_2xrts1,origin_param,speed_param,Nsamples,lt_barrier=lt_time_rts1,rt_barrier=rt_time_rts1rts1,suppress=suppress))


def get_strand_pu(r):
    r = np.array(r)
    top = r[:,0]
    bottom = r[:,1]

    ef = (top+1)/2.
    df = 1 - ef
    er = (bottom+1)/2.
    dr = 1 - er

    return ef,df,er,dr


