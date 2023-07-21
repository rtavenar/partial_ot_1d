import numpy as np
from scipy.spatial.distance import cdist


def update_points(points_in_x, neighbors, pair):
    """
    Examples
    --------
    >>> x = [1, 2, 0]
    >>> neighbors = np.array([0, 2, 0], dtype=int)
    >>> update_points(x, neighbors, [1, 0])
    []
    >>> update_points([8, 10, 12, 11, 2, 7, 1, 9, 3, 6, 0, 4, 5], np.array([ 4,  4,  4,  4,  4,  4,  5,  5,  5,  6,  8, 10, 12], dtype=int), [8, 5])
    [10, 12, 11, 2, 1, 9, 3, 0, 4, 5]
    """
    i_x, j_y = pair
    try:
        idx_x = points_in_x.index(i_x)
        del points_in_x[idx_x]
    except ValueError:
        pass  # the point is no longer in the list
    try:
        indices_y = np.where(neighbors == j_y)[0]
        for idx_y in indices_y:
            if idx_y in points_in_x:
                idx_x = points_in_x.index(idx_y)
                del points_in_x[idx_x]
    except IndexError:
        pass  # the point is no longer in the neighbors
    return points_in_x


def compute_cost_for_bunch_of_clusters(clusters, cumsum_mat):
    s = 0.
    for c in clusters:
        i0, i1, j0, j1 = c  # slice [i0, i1] in x matches slice [j0, j1] in y
        s += cumsum_mat[i1, j1] - cumsum_mat[i0, j0]
    return s


def expand_cluster(clusters, idx_c, cumsum_mat, new_i, new_j):
    # Here one can assume that:
    # * new_i is either i0 - 1 or i1 + 1
    # * new_j is either j0 - 1 or j1 + 1
    i0, i1, j0, j1 = clusters[idx_c]  # slice [i0, i1] in x matches slice [j0, j1] in y
    assert new_i in [i0 - 1, i1 + 1] and new_j in [j0 - 1, j1 + 1]
    cost_before = cumsum_mat[i1, j1] - cumsum_mat[i0, j0]
    # Case 1: new_i < 0 or new_i > n - 1 or new_j < 0 or new_j > n - 1: return inf
    # Else:
    clusters_to_be_updated = [clusters[idx_c]]
    new_clusters = None  # TODO
    
    return compute_cost_for_bunch_of_clusters(new_clusters) - compute_cost_for_bunch_of_clusters(clusters_to_be_updated)
    # TODO: return more info that we might need in the end?
    



def update_clusters(clusters, pair):
    i, j = pair
    new_pair_is_inserted = False
    for idx_c in range(len(clusters)):
        i0, i1, j0, j1 = clusters[idx_c]  # slice [i0, i1] in x matches slice [j0, j1] in y
        if (i < i0 - 1) or (j < j0 - 1):
            # Then the pair forms a new cluster since it has not been
            # merged with a previous cluster and it comes before the current one
            # with no possibility to merge it to the current one
            # and clusters are assumed to be sorted
            clusters.insert(idx_c, [i, i, j, j])
            new_pair_is_inserted = True
            break
        elif (i > i1 + 1) or (j > j1 + 1):
            # Then the pair cannot be merged to the current cluster, it comes 
            # strictly after it, let's see if it can be merged to a later cluster then
            pass
        else:
            # We can merge the points into the current cluster
            clusters[idx_c] = [min(i0, i), max(i1, i), min(j0, j), max(j1, j)]
            new_pair_is_inserted = True
            break
    if not new_pair_is_inserted:
        # We should insert at the end, then
        clusters.append([i, i, j, j])
    return clusters


def partial_ot_1d(x, y):
    n = len(x)
    assert len(y) == n, "For now"
    clusters = []
    previous_cost = 0.
    cdist_mat = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), 'sqeuclidean')  # O(n^2)
    cumsum_mat = np.cumsum(cdist_mat).reshape(cdist_mat.shape)
    x_neighbors = cdist_mat.argmin(axis=1)
    print(x_neighbors)
    x_neighbor_distances = cdist_mat[np.arange(n), x_neighbors]
    print(x_neighbor_distances)
    sorted_points_in_x = list(x_neighbor_distances.argsort())
    print(sorted_points_in_x)
    print("DONE")
    solutions = []
    costs = []
    for m in range(1, n):
        potential_new_costs = []
        potential_new_points = []
        # 1. check for brand new pairs of points to be added
        # if such a pair is to be added, it should correspond to the first
        # element in sorted_points_in_x since we removed useless pairs at 
        # the end of the previous iteration
        for i in sorted_points_in_x:
            j = x_neighbors[i]
            potential_new_costs.append(previous_cost + cdist_mat[i, j])
            potential_new_points.append([i, j])
            break

        # 2. check if clusters could be augmented
        # TODO: need to deal with possible cluster merges
        for idx_c, c in enumerate(clusters):
            i0, i1, j0, j1 = c  # slice [i0, i1] in x matches slice [j0, j1] in y
            cost_c = cumsum_mat[i1, j1] - cumsum_mat[i0, j0]
            
            if j1 < n - 1 and i0 > 0:
                cost_left_right = cumsum_mat[i1, j1 + 1] - cumsum_mat[i0 - 1, j0]
            else:
                cost_left_right = np.inf
            
            if i1 < n - 1 and j0 > 0:
                cost_right_left = cumsum_mat[i1 + 1, j1] - cumsum_mat[i0, j0 - 1]
            else:
                cost_right_left = np.inf
            if i0 > 0 and j0 > 0:
                cost_left_left = cumsum_mat[i1, j1] - cumsum_mat[i0 - 1, j0 - 1]
            else:
                cost_left_left = np.inf
            if i1 < n - 1 and j1 < n - 1:
                cost_right_right = cumsum_mat[i1 + 1, j1 + 1] - cumsum_mat[i0, j0]
            else:
                cost_right_right = np.inf
            print(cost_left_right - cost_c, 
                  cost_right_left - cost_c, 
                  cost_left_left - cost_c, 
                  cost_right_right - cost_c)
            if cost_left_right < min(cost_right_left, cost_left_left, cost_right_right):
                potential_new_costs.append(previous_cost + cost_left_right - cost_c)
                potential_new_points.append([i0 - 1, j1 + 1])
            elif cost_right_left < min(cost_left_right, cost_left_left, cost_right_right):
                potential_new_costs.append(previous_cost + cost_right_left - cost_c)
                potential_new_points.append([i1 + 1, j0 - 1])
            elif cost_left_left < min(cost_left_right, cost_right_left, cost_right_right):
                potential_new_costs.append(previous_cost + cost_left_left - cost_c)
                potential_new_points.append([i0 - 1, j0 - 1])
            else:
                potential_new_costs.append(previous_cost + cost_right_right - cost_c)
                potential_new_points.append([i1 + 1, j1 + 1])
            print(potential_new_costs[-1], potential_new_points[-1])

        # 3. decide on the best solution
        new_solution = np.argmin(potential_new_costs)
        print(f"m={m}", new_solution)
        previous_cost = potential_new_costs[new_solution]
        print("CLUSTERS BEFORE: ", clusters)
        clusters = update_clusters(clusters, potential_new_points[new_solution])
        # update sorted_points_in_x
        # using information stored in potential_new_points[new_solution]
        print("POINTS BEFORE: ", sorted_points_in_x)
        print("PAIR: ", potential_new_points[new_solution])
        sorted_points_in_x = update_points(sorted_points_in_x, x_neighbors, potential_new_points[new_solution])
        print("POINTS AFTER: ", sorted_points_in_x)
        print("CLUSTERS AFTER: ", clusters)
        solutions.append(list(clusters))
        costs.append(previous_cost)
    return solutions
