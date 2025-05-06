import copy
import torch
import random
import numpy as np


def cnt_node_occurrence(edge_index):
    """
    This function is used to count the occurrence of each node in the edge_index.
    """
    node_occurrence = {}
    for node in edge_index[0]:
        if node.item() not in node_occurrence:
            node_occurrence[node.item()] = 1
        else:
            node_occurrence[node.item()] += 1
    return node_occurrence

def find_first_two_indices(tensor, aug_ratio, node_occurrence, node_indices):
    """
    This function is used to find the first two indices of the tensor that can be removed.
    """
    indices_dict = {}
    result = []
    
    for i, element in enumerate(tensor):
        hyperedge_id = element.item()
        if hyperedge_id not in indices_dict:
            indices_dict[hyperedge_id] = [i]
        else:
            indices_dict[hyperedge_id].append(i)
    
    for indices in indices_dict.values():
        edge_len = len(indices)  
        if edge_len > 2:
            num_to_remove = int(aug_ratio * edge_len)  
            if num_to_remove < 1:
                continue  
            sample_indices = np.random.permutation(indices) 
            removed = 0
            for idx in sample_indices: 
                node_id = node_indices[idx].item()  
                if node_occurrence[node_id] > 1:
                    result.append(idx)  
                    node_occurrence[node_id] -= 1  
                    removed += 1  
                    if removed >= num_to_remove:
                        break
                else:
                    continue 
    return result


def permute_edges(ni, index, aug_ratio):
    """
    Augment hyperedges by permuting the nodes in the hyperedges.
    """
    add_e = False
    edge_index = copy.deepcopy(index)
    node_num = int(edge_index[0].max() + 1)
    
    node_occurrence = cnt_node_occurrence(edge_index)
    
    remove_index = find_first_two_indices(edge_index[1], aug_ratio, node_occurrence, edge_index[0])
    
    keep_index = [idx for idx in range(len(edge_index[1])) if idx not in remove_index]
    edge_after_remove1 = edge_index[:, keep_index]
    
    if add_e:
        permute_num = int(len(remove_index))
        idx_add_1 = np.random.choice(node_num, permute_num)
        idx_add = np.stack((idx_add_1, edge_index[1][remove_index].cpu()), axis=0)
        edge_index = torch.tensor(np.concatenate((edge_after_remove1.cpu(), idx_add), axis=1))
    else:
        edge_index = edge_after_remove1

    indices_dict = {}
    existing_node_sets = set()
    for i in edge_index[1].unique():
        i = i.item()
        node_indices = set(edge_index[0][edge_index[1] == i].tolist())
        node_set = frozenset(node_indices)
        while node_set in existing_node_sets:
            add = np.random.choice(node_num, 1).item()  
            node_indices.add(add)
            node_set = frozenset(node_indices)
        existing_node_sets.add(node_set)
        indices_dict[i] = list(node_indices)
    return indices_dict
