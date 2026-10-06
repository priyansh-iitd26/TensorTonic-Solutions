def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """

    precision_recall_k_lst = [0.0] * 2

    relevant_set = set(relevant)

    counter = 0
    
    for recommendation in recommended[:k]:
        if recommendation in relevant_set:
            counter += 1
    
    precision_recall_k_lst[0] = counter / k
    precision_recall_k_lst[1] = counter / len(relevant_set)

    return precision_recall_k_lst