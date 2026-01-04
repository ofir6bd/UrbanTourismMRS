import numpy as np
from numba import njit, i8, f8
from utils import argtopk, haversine

@njit(
    f8[::1](i8, i8[::1])
)
def do_planning(  # --> define {+0,+1} sustainable promotion utility
        J,
        croi,
):
    advt_preferences = np.ones(J, dtype="f8")
    advt_preferences[croi] = 0
    return advt_preferences

@njit(
    i8[::1](f8[::1], f8[::1], f8[:,::1], f8[::1], i8, f8, f8)
)
def recommend_with_void_search_decoder(
        user_preferences,             
        advt_preferences,             
        locations,                    
        user_location,                
        pack_size,                    
        lamb,                         
        distance_penalty_weight,      
):
    assert (0 <= lamb <= 1), "Lambda is out of range!"

    # estimate gumbel standard deviation
    _RERANKING_BETA = 0.001 * (6**0.5 / np.pi)

    # estimate gumbel noise
    _r = _RERANKING_BETA * (-1) * np.log(-np.log( np.random.random(size=len(user_preferences)) ))
    _s = _RERANKING_BETA * (-1) * np.log(-np.log( np.random.random(size=len(user_preferences)) ))

    # calculate distances to all POIs
    distances = np.zeros(len(user_preferences), dtype="f8")
    for j in range(len(user_preferences)):
        distances[j] = haversine(user_location, locations[j,:])
    
    # normalize distances from 0 to 1
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    if max_dist > min_dist:
        normalized_distances = (distances - min_dist) / (max_dist - min_dist)
    else:
        normalized_distances = np.zeros(len(distances), dtype="f8")

    # apply distance penalty to user preferences
    penalized_user_preferences = user_preferences - distance_penalty_weight * normalized_distances
    
    # multistakeholder recommendation with stochastic re-ranking to break utility ties
    pack = argtopk((1-lamb) * (penalized_user_preferences+_r) + lamb * (advt_preferences+_s), k=pack_size)

    return pack
