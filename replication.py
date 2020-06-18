"""
ecam Jul19
"""

import multiprocessing as mp #Parallel execution
from operator import add #Reduce operation
import random #Pseudo-random number generator
import numpy as np #Summable arrays

from math import ceil #Trying to use std library.


#Origin sorting
def sort_origins_by_loc(origins):
    """
    Args:
        origins list of origin pairs [ [location,activation time], ... ]

    Returns:
        list of origin pairs, sorted by location.
    """

    return sorted(origins,key=lambda x:x[0])

def sort_origins_by_time(origins):
    """
    Args:
        origins list of origin pairs [ [location,activation time], ... ]

    Returns:
        list of origin pairs, sorted by time
    """

    return sorted(origins,key=lambda x:x[1])


def get_replication_time(origin,x,lt_time=lt_time,rt_time=rt_time,speed_params=[],chunk=[]):
    """
    The computation assumes a fork at location and time defined by origin, moving towards x.

    Args:
        origin origin (or starting positions and time) of the for to replicate x
        x location that we are timing
        lt_time function to compute the left travelling replication time
        rt_time function to cmopute the right travelling replication time

    Returns:
        Time at which the origin replicates the location x
    """

    #Replication to the right
    if origin[0] < x:
        return rt_time(origin,x,speed_params)

    #Replication to the left
    if origin[0] > x:
        return lt_time(origin,x,speed_params)

    #x is actually the origin
    return origin[1] 

def find_first_origin(origins,lt_time=lt_time,rt_time=rt_time,speed_params=[]):
    """
    Finds the first origin to fire in a list, and removes passively replicated origins.

    Args:
        origins list of origin pairs, sorted by time.

    Returns:
        a triplet with the first origin in the list that will fire, 
        and lists of active origins to the left and right of it.
    """

    o = origins[0]

    #Only one origin
    if len(origins) == 1:
        return o,[],[]

    L = []
    R = []

    #Fill lists
    for oo in origins[1:]:
        tau = get_replication_time(o,oo[0],lt_time,rt_time,speed_params)

        #o fires LATER than replication time. It will not fire.
        if oo[1] >= tau:
            continue

        #Otherwise, add o to list
        if oo[0] < o[1]:
            L.append(oo)
        else:
            R.append(oo)

    return o,L,R

def find_active_origins(origins,lt_time=lt_time,rt_time=rt_time,speed_params=[]):
    """
    Args:
        origins list of origin pairs, sorted by time.

    Returns:
        list of active origins, sorted by location.
    """
    
    #No more origins to process    
    if len(origins) == 0:
        return []

    r = find_first_origin(origins,lt_time,rt_time,speed_params) #First firing origin

    L = find_active_origins(r[1],lt_time,rt_time,speed_params) #Active to the left of first firing one
    R = find_active_origins(r[2],lt_time,rt_time,speed_params) #same, to the right

    return L + [ r[0] ] + R

def find_termination_ab(oa,ob,lt_time=lt_time,rt_time=rt_time,speed_params=[]):
    """
    Bisection to find termination location between origins a and b
    Returns pair termination x, termination time

    Args:
        oa left replication origin
        ob right replication origin
    
    Returns:
        A pair [x,t] with the location and time of termination. 

    Note:
        assumes location oa < location ob
        x can be outside the grid (between nodes)
        when x is between nodes, t is set to max time of nearest two nodes.
    """

    l = oa[0] #Left bound
    r = ob[0] #Right bound

    while r - l > 1:
        term = (r+l) / 2
        
        if get_replication_time(ob,term,lt_time,rt_time,speed_params) \
                    < get_replication_time(oa,term,lt_time,rt_time,speed_params):
            r = term
        else:
            l = term
    
    #Check: termination at node
    t = get_replication_time(ob,l,lt_time,rt_time,speed_params) 
    tl = get_replication_time(oa,l,lt_time,rt_time,speed_params)

    #Left and right forks arrive at same time
    if t == tl:
        return [l,tl]

    #Check: termination at different times
    tr = get_replication_time(ob,r,lt_time,rt_time,speed_params)
    
    #Most common case (no barrier, symmetric term)
    if tr == tl:
        return [float(r+l)/2,tl]

    t2 = get_replication_time(oa,r,lt_time,rt_time,speed_params)

    #Same time to reach r
    if t2 == tr:
        return [r,tr]

    #Any other case we have off termination. 
    return [float(r+l)/2,max([tl,tr])]

def find_terminations(origins,lt_time=lt_time,rt_time=rt_time,speed_params=[]):
    """
    input: active origins sorted by loc
    output: terminations pairs, note that x are floats in there!
    """
    #one origin or less: no terminations
    if len(origins) <= 1:
        return []            

    return [find_termination_ab(o1,o2,lt_time,rt_time,speed_params) for o1,o2 in zip(origins[:-1],origins[1:]) ]

def get_chunks(active,term,L,lt_time=lt_time,rt_time=rt_time,speed_params=[]):
    """
    Args:
        active active origins, sorted by loc
        term terminations, sorted by loc
        L total number of bins/positions/locations, i.e. length of the space
        lt_time,rt_time,speed_params as before.

    Returns:
        list of chunks [ termination, origin, termination ], corresponding to the activity of origin.
    """

    #Boundary terminations 
    left_termination = [0, get_replication_time(active[0],0,lt_time,rt_time,speed_params) ]
    right_termination = [L-1, get_replication_time(active[-1],L-1,lt_time,rt_time,speed_params) ]

    #Only one origin
    if len(active) == 1:
        return [ [ left_termination, active[0], right_termination ] ]

    #Any other case, a chunk per origin.
    res = [None] * len(active) #One chunk per origin

    res[0] = [left_termination,active[0],term[0] ]

    for i,x in enumerate(zip(term[:-1],active[1:-1],term[1:])):
        res[i+1] = x
        
    res[-1] = [term[-1],active[-1],right_termination]

    return res

def process_chunks(chunk_list,chunk_processor,L,lt_time,rt_time,speed_params):
    """
    Args:
        chunk_list list of [term,origin,term] chunks
        chunk_processor has the signature of get_replication_time, and gives the values of the chunk.
        L length of space
        lt,rt,speed as before
    """
    r = [None]*L

    for chunk in chunk_list:
        #Start AFTER off node term, or AT on node term.
        #End BEFORE end node term, or BEFORE on node term.
        start = int(ceil(chunk[0][0]))
        end = int(ceil(chunk[2][0]))

        #From termination to termination 
        for i in xrange(start,end): 
            if isinstance(r[i],type(None)):
                r[i] = chunk_processor(chunk[1],i,lt_time,rt_time,speed_params,chunk)
            else: 
                r[i] = r[i] + chunk_processor(chunk[1],i,lt_time,rt_time,speed_params,chunk)
            #ct[i] = ct[i]+1

        #Value at last termination, processed manually
        if end == L-1:
            if isinstance(r[end],type(None)):
                r[end] = chunk_processor(chunk[1],end,lt_time,rt_time,speed_params,[]) 
            else:
                r[end] = r[end] + chunk_processor(chunk[1],end,lt_time,rt_time,speed_params,[]) 
    return np.array(r)

def get_direction(o,x,lt_time,rt_time,params,chunk):
    """
    Returns in what direction origin o replicates x

    Args:
        o origin pair (location,time)
        x location to look at
        lt,rt,params as before

    Returns:
        1 (replicated to the right), -1 (replicated to the left)
    """
    #x is an origin
    if o[0] == x:
        return 0

    #left bdry
    if x == 0:
        return -1

    #Check for on node termination
    if len(chunk) > 0:
        if chunk[0][0] == x or chunk[2][0] == x:
            return 0

    if o[0] < x:
        return 1

    return -1

def process_time_chunks(chunk_list,L,lt_time,rt_time,speed_params):
    """
    Args:
        chunk_list list of [term,origin,term] chunks
        L length of space
        lt,rt,speed as before
    """
    return process_chunks(chunk_list,get_replication_time,L,lt_time,rt_time,speed_params)

def process_direction_chunks(chunk_list,L,lt_time,rt_time,speed_params):
    return process_chunks(chunk_list,get_direction,L,lt_time,rt_time,speed_params)

def process_time_chunks_star(p):
    return process_time_chunks(*p)

def process_direction_chunks_star(p):
    return process_direction_chunks(*p)

def process_chunks_star(p):
    return process_chunks(*p)

def find_active_origins_star(p):
    return find_active_origins(*p)

def find_terminations_star(p):
    return find_terminations(*p)

def get_chunks_star(p):
    return get_chunks(*p)

def do_experiment_serial(L,get_origins,get_origins_params,lt_time,rt_time,speed_params,chunk_processor,samples):
    """
    Args:
        L length of space
        get_origins origin sampling function
        get_origins_params params for get_origins
        lt,rt,speed_params as before
        chunk_processor function to process chunks (e.g. for time or direction)
        samples number of samples in the experiment

    Returns:
        Mean of the experiment
    """

    #Generate origins
    origin_list = map(get_origins,[ get_origins_params for i in xrange(samples) ] )
    #Active origins 
    active_list = map(find_active_origins_star, \
                      zip(origin_list,[lt_time]*samples,[rt_time]*samples,[speed_params]*samples) )
    #Terminations

    #Sort
    active_list = map(sort_origins_by_loc,active_list)

    term_list = map(find_terminations_star, \
                    zip(active_list,[lt_time]*samples,[rt_time]*samples,[speed_params]*samples) )
    #Chunks 
    chunk_list = map(get_chunks_star, \
                     zip(active_list,term_list,[L]*samples,[lt_time]*samples,[rt_time]*samples,[speed_params]*samples) ) 
    #Join chunks in one list of chunks (insted of list of lists
    chunks = reduce(add,chunk_list)
    #Process chunks (serial)
    r = process_chunks(chunks,chunk_processor,L,lt_time,rt_time,speed_params)    
    return r / float(samples)


