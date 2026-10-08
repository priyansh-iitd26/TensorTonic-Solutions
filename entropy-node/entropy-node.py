def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    
    if len(y) == 0:
        return 0.0

    # correctly count the occurrences of each class label
    seen = {}
    
    for class_label in y:
        if class_label not in seen:
            seen[class_label] = 1
        else:
            seen[class_label] += 1

    entropy = 0.0
    total_elements = len(y)

    for count in seen.values():
        probability = count / total_elements
        entropy -= probability * np.log2(probability)

    return entropy

    