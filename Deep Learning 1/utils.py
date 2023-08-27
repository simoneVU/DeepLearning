def flatten(list_to_flatten):
    flattened_list = [item for sublist in list_to_flatten for item in sublist]
    return flattened_list

def normalize(list_values):
    list_values_f = flatten(list_values)
    min_value = min(list_values_f)
    max_value = max(list_values_f)
    for value in list_values:  
        for i in range(len(value)):      
            value[i] = ((value[i] - min_value)/(max_value - min_value))
            value[i] = value[i]*(1 -(-1)) + -1
    return list_values