import numpy as np


def partial_ot(x,y):
    xy, lab_xy, pos_xy, dict_cum_sum_lab_x, cum_sum_costs_x_repeated, dict_cum_sum_lab_y, cum_sum_costs_y_repeated, cum_sum_lab_xy, cum_sum_costs_x, cum_sum_costs_y, cost_groups, cost_series = init_useful_vectors(x, y)
    #print(lab_xy)
    #print("pos", pos_xy)
    dict_candidates, dict_candidates_cost = compute_costs_for_reg_path(xy, lab_xy, dict_cum_sum_lab_x, cum_sum_costs_x_repeated, dict_cum_sum_lab_y, cum_sum_costs_y_repeated, cum_sum_lab_xy, cum_sum_costs_x, cum_sum_costs_y, cost_groups, cost_series)
    #print(dict_candidates_cost)
    return compute_reg_path(dict_candidates, dict_candidates_cost, xy, lab_xy, pos_xy, cost_series, cost_groups)



def init_useful_vectors(x, y):

    #labels and positions
    lab_x = np.ones(len(x), dtype=int)
    lab_y = -1 * np.ones(len(y), dtype=int)
    pos_x = np.arange(len(x))
    pos_y = -1 * np.arange(len(x))

    #concatenated sorted distributions and sorted labels
    xy = np.concatenate((x,y))
    r_xy = np.argsort(xy)
    xy = xy[r_xy]
    lab_xy = np.concatenate((lab_x, lab_y))
    lab_xy = lab_xy[r_xy]
    pos_xy = np.concatenate((pos_x, pos_y))
    pos_xy = pos_xy[r_xy]
    print("positions", pos_xy)

    #cumulated sum over the sorted concatenated labels and costs
    cum_sum_lab_xy = np.cumsum(lab_xy)
    cum_sum_costs_x = np.cumsum(x)
    cum_sum_costs_y = np.cumsum(y)
    cum_sum_costs_xy = np.concatenate((cum_sum_costs_x, cum_sum_costs_y))
    cum_sum_costs_xy = cum_sum_costs_xy[r_xy]

    #cumulated sum with constant values
    cum_sum_costs_x_repeated = np.zeros(len(xy))
    cum_sum_costs_y_repeated = np.zeros(len(xy))

    #dictionnaries of unique cum_sum_lab_xy, per labels
    keys_x = np.unique(cum_sum_lab_xy)
    keys_x = np.concatenate(([np.min(keys_x)-1], keys_x, [np.max(keys_x)+1]))
    keys_y = np.unique(cum_sum_lab_xy)
    keys_y = np.concatenate(([np.min(keys_y)-1], keys_y, [np.max(keys_y)+1]))  
    dict_cum_sum_lab_x = dict.fromkeys(keys_x)
    dict_cum_sum_lab_y = dict.fromkeys(keys_y)

    #initialize cost vectors
    cost_series = np.zeros(len(xy))
    cost_groups = np.empty(len(xy))

    return xy, lab_xy, pos_xy, dict_cum_sum_lab_x, cum_sum_costs_x_repeated, dict_cum_sum_lab_y, cum_sum_costs_y_repeated, cum_sum_lab_xy, cum_sum_costs_x, cum_sum_costs_y, cost_groups, cost_series


def compute_costs_for_reg_path(xy, lab_xy, dict_cum_sum_lab_x, cum_sum_costs_x_repeated, dict_cum_sum_lab_y, cum_sum_costs_y_repeated, cum_sum_lab_xy, cum_sum_costs_x, cum_sum_costs_y, cost_groups, cost_series):
    dict_candidates = {}
    dict_candidates_cost = {}

    #initialization
    if lab_xy[0] == 1:
        dict_cum_sum_lab_x[1] = 0
        cum_sum_costs_x_repeated[0] = xy[0]
    else:
        dict_cum_sum_lab_y[-1] = 0
        cum_sum_costs_y_repeated[0] =  xy[0]


    for i in range(1, len(xy)):
        current_cum_sum = cum_sum_lab_xy[i]
        #print("---------current cum sum", current_cum_sum)
        if lab_xy[i] == 1:
            #update cum sum repeated
            cum_sum_costs_x_repeated[i] = cum_sum_costs_x_repeated[i-1] + xy[i]
            cum_sum_costs_y_repeated[i] = cum_sum_costs_y_repeated[i-1]
            #update dictionnary of sample of idx i
            dict_cum_sum_lab_x[current_cum_sum] = i
            # look if there is a possible group within the previous samples
            if dict_cum_sum_lab_y[current_cum_sum - 1] is not None:
                prev_cum_sum = dict_cum_sum_lab_y[current_cum_sum-1] #get the index of the beginning of the group
                dict_cum_sum_lab_y[current_cum_sum-1] = None # remove the index from the dictionnary as there can be only one correpondance
                cost_groups[i] = np.abs(+ cum_sum_costs_x_repeated[i] - cum_sum_costs_x_repeated[prev_cum_sum] - cum_sum_costs_y_repeated[i] + cum_sum_costs_y_repeated[prev_cum_sum-1]) # compute the cost of the group
                if prev_cum_sum - 1 > 0: #if we are not at the beginning of the distributions
                    cost_series[i] = cost_series[prev_cum_sum - 1] + cost_groups[i]
                else:
                    cost_series[i] = cost_groups[i]
                            #update the dict_candidates if the groups contains 2 points
                if (i - prev_cum_sum) == 1:
                    dict_candidates[i] = prev_cum_sum
                    dict_candidates_cost[i] = cost_groups[i]

        else: #same thing but change the vectors
            #update cum sum repeated
            cum_sum_costs_y_repeated[i] = cum_sum_costs_y_repeated[i-1] + xy[i]
            cum_sum_costs_x_repeated[i] = cum_sum_costs_x_repeated[i-1]
            #update dictionnary of sample of idx i
            dict_cum_sum_lab_y[current_cum_sum] = i
            # look if there is a possible group within the previous samples
            if dict_cum_sum_lab_x[current_cum_sum + 1] is not None:
                prev_cum_sum = dict_cum_sum_lab_x[current_cum_sum+1] #get the index of the beginning of the group
                dict_cum_sum_lab_x[current_cum_sum+1] = None # remove the index from the dictionnary as there can be only one correpondance
                cost_groups[i] = np.abs(- cum_sum_costs_y_repeated[i] + cum_sum_costs_y_repeated[prev_cum_sum] + cum_sum_costs_x_repeated[i] - cum_sum_costs_x_repeated[prev_cum_sum-1]) # compute the cost of the group
                if prev_cum_sum - 1 > 0: #if we are not at the beginning of the distributions
                    cost_series[i] = cost_series[prev_cum_sum - 1] + cost_groups[i]
                else:
                    cost_series[i] = cost_groups[i]
                #update the dict_candidates if the groups contains 2 points
                if (i - prev_cum_sum) == 1:
                    dict_candidates[i] = prev_cum_sum
                    dict_candidates_cost[i] = cost_groups[i]
    print("cost groups then series")
    print(cost_groups)
    print(cost_series)
    return dict_candidates, dict_candidates_cost



def compute_reg_path(dict_candidates, dict_candidates_cost, xy, lab_xy, pos_xy, cost_series, cost_groups):
    current_series_idx = {} #the series that are in the active set
    current_series_cost = {} #and their associated costs
    active_x = np.zeros(len(xy) // 2)
    active_y = np.zeros(len(xy) // 2)
    active_cost = np.zeros(len(xy) // 2)

    for i in range(len(xy)//2):
        #take the candidate that has the smallest **marginal** cost
        idx_end = min(dict_candidates_cost, key=dict_candidates_cost.get) 
        idx_end_init = idx_end #to keep in memory
        idx_start = dict_candidates[idx_end]
        idx_start_init = idx_start

        #update the active set
        if pos_xy[idx_end] > 0:
            active_x[i] = pos_xy[idx_end]
            active_y[i] = pos_xy[idx_start]
        else:
            active_y[i] = pos_xy[idx_end]
            active_x[i] = pos_xy[idx_start]       

        #--------------update the list of current series (i.e. merge some series)
        if (idx_end + 1) in current_series_idx.values(): #2series have to be merged
            current_key = list(current_series_idx.keys())[list(current_series_idx.values()).index(idx_end + 1)] 
            idx_end = current_key

        if (idx_start - 1) in current_series_idx: #2series have to be merged
            idx_start = current_series_idx[idx_start - 1]
            del current_series_idx[idx_start_init - 1]
            del current_series_cost[idx_start_init - 1]

        current_series_idx[idx_end] = idx_start # add the series within the current active set
        current_series_cost[idx_end] = cost_series[idx_end]
        if idx_start > 0:
             current_series_cost[idx_end] -= cost_series[idx_start - 1]

        del dict_candidates[idx_end_init]# remove the candidate
        del dict_candidates_cost[idx_end_init]
        
        #--------------update the current series if needed (eg. if start - end has been added, start +1 -- end -1 should be removed)
        if (idx_end_init-1) in current_series_idx:
            if current_series_idx[idx_end_init-1] == idx_start_init+1:
                del current_series_idx[idx_end_init - 1]
                del current_series_cost[idx_end_init - 1]
        if idx_start_init in dict_candidates: #if we have added 7-10, couple 6-7 is not among the candidates anymore
            del dict_candidates[idx_start_init]
            del dict_candidates_cost[idx_start_init]
        if idx_end_init in dict_candidates.values(): #if we have added 7-10, couple 10-11 is not among the candidates anymore
            tmp = list(dict_candidates.keys())[list(dict_candidates.values()).index(idx_end_init)] 
            del dict_candidates[tmp]
            del dict_candidates_cost[tmp]


        #--------------and add potentially new candidates for the next step
        if idx_end < (len(xy) - 1) and idx_start > 0: #if it is possible 
            if lab_xy[idx_end + 1] != lab_xy[idx_start - 1]: #if the two extreme points have different labels, we add them on the candidate set
                dict_candidates[idx_end+1] = idx_start - 1
                dict_candidates_cost[idx_end+1] = cost_series[idx_end+1] - cost_series[idx_end] + cost_series[idx_start-1]
                if idx_start > 1:
                    dict_candidates_cost[idx_end+1] -= cost_series[idx_start-2]

        active_cost[i] = np.sum(list(current_series_cost.values()))/len(xy) * 2
        
    return current_series_idx, current_series_cost, active_x, active_y, active_cost

