import numpy as np
from copy import deepcopy
import time

EXT_GROUP_TYPE_1 = 1
EXT_GROUP_TYPE_2 = 2
NEW_PAIR = 3


class Group:
    """A Group is a set of consecutive pairs.
    In a group:
    * `i_x` is the first index from x
    * `i_y` is the first index from y
    * `length` is the number of consecutive pairs in the group
    * `j_x` is the last included index from x (computed on the fly)
    * `j_y` is the last included index from y (computed on the fly)
    * `pairs` is the ordered list of all pairs in the group (computed on the fly)
    """
    def __init__(self, i_x, i_y, length):
        self.i_x = i_x
        self.i_y = i_y
        self.length = length
    
    @property
    def j_x(self):
        return self.i_x + self.length - 1
    
    @property
    def j_y(self):
        return self.i_y + self.length - 1
    
    @property
    def as_tuple(self):
        return (self.i_x, self.i_y, self.length)
    
    @property
    def pairs(self):
        return [(self.i_x + delta_i, self.i_y + delta_i) 
                for delta_i in range(self.length)]
    
    @property
    def first_pair(self):
        return (self.i_x, self.i_y)
    
    @property
    def last_pair(self):
        return (self.i_x + self.length - 1, self.i_y + self.length - 1)
    
    def extend(self, type):
        if type == EXT_GROUP_TYPE_1:
            self.i_x -= 1
        else:
            self.i_y -= 1
        self.length += 1
    
    def touches(self, other):
        """`a.touches(b)` is True iff `a` and `b` are disjoint 
        and could be merged to form a new group of consecutive pairs.

        >>> g0 = Group(4, 3, 2)
        >>> g1 = Group(6, 5, 3)
        >>> g2 = Group(9, 9, 4)
        >>> g0.touches(g1)
        True
        >>> g1.touches(g0)
        True
        >>> g0.touches(g2)
        False
        >>> g2.touches(g0)
        False
        >>> g2.touches(g1)
        False
        >>> g1.touches(g2)
        False
        """
        if other.i_x < self.i_x:
            return other.touches(self)
        return (self.i_x + self.length == other.i_x and
                self.i_y + self.length == other.i_y)
    
    def intersects(self, pair):
        i_x, i_y = pair
        return ((self.i_x <= i_x <= self.j_x) or 
                (self.i_y <= i_y <= self.j_y))
        
    def __eq__(self, other):
        return (self.i_x == other.i_x and 
                self.i_y == other.i_y and 
                self.length == other.length)
    
    def __hash__(self):
        return hash(self.as_tuple)
    
    def __add__(self, other):
        """
        `a + b` is the result of the merge of `a` and `b` into a new group.
        This operation is valid iff `a.touches(b)`.

        >>> g0 = Group(4, 3, 2)
        >>> g1 = Group(6, 5, 3)
        >>> g0.touches(g1)
        True
        >>> g0 + g1
        <Group x_span=(4, 8), y_span=(3, 7)>
        """
        return Group(
            i_x=min(self.i_x, other.i_x),
            i_y=min(self.i_y, other.i_y),
            length=self.length + other.length
        )
    
    def __str__(self):
        # Spans are understood as including bounds
        return f"<Group x_span=({self.i_x}, {self.j_x}), y_span=({self.i_y}, {self.j_y})>"
    
    def __repr__(self):
        return str(self)

def argmin(l, key=None):
    """Return position of the minimum in the list 
    (applying function `key` if providing to each element).
    
    >>> l = [4, 5, 3, 2, 7]
    >>> argmin(l)
    3
    >>> l = [(0, 5), (3, 2), (2, 8)]
    >>> argmin(l, key=lambda t: t[1])
    1
    """
    if key is None:
        return np.argmin(l)
    return np.argmin([key(li) for li in l])


def is_possible_type_1(groups, idx, n):
    g = groups[idx]
    if g.i_x == 0 or g.j_y == n - 1:
        return False
    if idx > 0 and groups[idx - 1].j_x == g.i_x - 1:
        return False
    if idx < len(groups) - 1 and groups[idx + 1].i_y == g.j_y + 1:
        return False
    # TODO: il y a aussi peut être un test supplémentaire (utile ?) pour voir si le coût n'est pas trop grand (> 2lambda)
    return True


def is_possible_type_2(groups, idx, n):
    g = groups[idx]
    if g.i_y == 0 or g.j_x == n - 1:
        return False
    if idx > 0 and groups[idx - 1].j_y == g.i_y - 1:
        return False
    if idx < len(groups) - 1 and groups[idx + 1].i_x == g.j_x + 1:
        return False
    return True


def get_diff_cost(extension_type, g, precomputed_costs_for_groups, x, y):
    new_group = deepcopy(g)
    new_group.extend(extension_type)
    if not new_group.as_tuple in precomputed_costs_for_groups.keys():
        cost = 0.
        # TODO: est ce qu'il y a moyen de faire mieux?
        for delta_i in range(new_group.length):
            i_x = new_group.i_x + delta_i
            i_y = new_group.i_y + delta_i
            cost += (x[i_x] - y[i_y]) ** 2
        precomputed_costs_for_groups[new_group.as_tuple] = cost
    return precomputed_costs_for_groups[new_group.as_tuple] - precomputed_costs_for_groups[g.as_tuple]


def get_insert_position(pair, groups):
    """Return the index at which the group corresponding to `pair` should be included in `groups`
    such that the order is maintained in `groups`.

    >>> g0 = Group(0, 0, 3)
    >>> g1 = Group(3, 5, 1)
    >>> g2 = Group(7, 7, 3)
    >>> p0 = (5, 6)
    >>> p1 = (8, 10)
    >>> get_insert_position(p0, [g0, g1, g2])
    2
    >>> get_insert_position(p1, [g0, g1, g2])
    3
    """
    i_x, i_y = pair
    for i_g, g in enumerate(groups):
        if g.i_x > i_x:
            return i_g
    return len(groups)


def _candidate_pairs(x, y):
    n = len(x)
    i_y = 0
    candidate_pairs = []
    for i_x in range(n):
        while i_y < n and y[i_y] < x[i_x]:
            i_y += 1
        if i_y < n:
            candidate_pairs.append((i_x, i_y))
        if i_y > 0:
            candidate_pairs.append((i_x, i_y-1))
    return candidate_pairs


def get_candidate_pairs(x, y):
    """Candidate pairs are pairs of the form (i_x, i_y) such that y[i_y] is either
    the nearest neighbour of x[i_x] from the left or the one from the right.
    """
    candidate_pairs_x_y = _candidate_pairs(x, y)
    candidate_pairs_y_x = [(i_x, i_y) for (i_y, i_x) in _candidate_pairs(y, x)]
    return list(set(candidate_pairs_x_y).intersection(set(candidate_pairs_y_x)))

def conflicts(pair, groups):
    for g in groups:
        if g.intersects(pair):
            return True
    return False


def partial_ot_1d(x, y, size_max=None):
    x.sort()
    y.sort()
    n = len(x)
    if size_max is None:
        size_max = n
    assert len(y) == n, "For now"
    groups = []
    history = []
    cost = 0.

    available_direct_pairs = get_candidate_pairs(x, y)
    cost_available_direct_pairs = {
        (i_x, i_y): (x[i_x] - y[i_y]) ** 2 for (i_x, i_y) in available_direct_pairs
    }
    available_direct_pairs = sorted(available_direct_pairs, 
                                    key=lambda k: cost_available_direct_pairs[k])

    precomputed_costs_for_groups = {}

    for i in range(1, size_max + 1):
        # Have done some profiling to understand where the most time is spent
        # A first conclusion is that the two parts that take most time are:
        # * Computing updated costs for group modifications
        # * Storing the list of groups at the end of each iter to be able to return the history later
        # In fact, `sum_n_groups.py` shows that \sum_i O(n_groups_at_iter_i) = O(n^2) so any operation 
        # that is O(n_groups) at each iter leads to overall complexity of O(n^2)
        possible_changes = []
        for i_g in range(len(groups)):
            if is_possible_type_1(groups, i_g, n):
                diff_cost = get_diff_cost(EXT_GROUP_TYPE_1, groups[i_g], precomputed_costs_for_groups, x, y)
                possible_changes.append((EXT_GROUP_TYPE_1, i_g, diff_cost))
            if is_possible_type_2(groups, i_g, n):
                diff_cost = get_diff_cost(EXT_GROUP_TYPE_2, groups[i_g], precomputed_costs_for_groups, x, y)
                possible_changes.append((EXT_GROUP_TYPE_2, i_g, diff_cost))

        if len(available_direct_pairs) > 0:
            idx_best_pair = 0
            next_available_pair = available_direct_pairs[idx_best_pair]
            cost_next_available_pair = cost_available_direct_pairs[next_available_pair]
            possible_changes.append((NEW_PAIR, next_available_pair, cost_next_available_pair))
        
        best_change = possible_changes[argmin(possible_changes, key=lambda t: t[2])]
        if best_change[0] == NEW_PAIR:
            idx_edit = get_insert_position(best_change[1], groups)
            g = Group(best_change[1][0], best_change[1][1], 1)
            groups.insert(idx_edit, g)
            precomputed_costs_for_groups[g.as_tuple] = best_change[2]
        else:
            groups[best_change[1]].extend(best_change[0])
            idx_edit = best_change[1]
        
        # We have to remove the first elements in `available_direct_pairs` if they are inside an existing group
        # Other alternative would be to remove all the elements in `available_direct_pairs` that conflict with the last
        # group modification (no need to parse all groups, but need to know where the pairs are in the list)
        # This step is negligible in the total computation time
        # n=1000, ~.06s spent in this part (to be compared to 1.77s for the whole algo)
        while len(available_direct_pairs) > 0 and conflicts(available_direct_pairs[0], groups):
            del available_direct_pairs[0]

        if idx_edit < len(groups) - 1 and groups[idx_edit].touches(groups[idx_edit + 1]):
            new_group = groups[idx_edit] + groups[idx_edit + 1]
            precomputed_costs_for_groups[new_group.as_tuple] = (
                precomputed_costs_for_groups[groups[idx_edit].as_tuple] + 
                precomputed_costs_for_groups[groups[idx_edit + 1].as_tuple]
            )
            groups[idx_edit] = new_group
            del groups[idx_edit + 1]
        if idx_edit > 0 and groups[idx_edit - 1].touches(groups[idx_edit]):
            new_group = groups[idx_edit - 1] + groups[idx_edit]
            precomputed_costs_for_groups[new_group.as_tuple] = (
                precomputed_costs_for_groups[groups[idx_edit - 1].as_tuple] + 
                precomputed_costs_for_groups[groups[idx_edit].as_tuple]
            )
            groups[idx_edit - 1] = new_group
            del groups[idx_edit]
        cost += best_change[2]
        history.append(deepcopy(groups))
    return history
