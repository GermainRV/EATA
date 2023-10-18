import numpy as np

def interval_categorizer(data, thresholds, category_labels=None, lower_endpoint=-np.inf, upper_endpoint=np.inf):
    if isinstance(thresholds, list): thresholds = np.array(thresholds)
    if (lower_endpoint == -np.inf) and (thresholds[0] != lower_endpoint): 
        thresholds = np.concatenate(([lower_endpoint], thresholds))
    if (upper_endpoint == np.inf) and (thresholds[-1] != upper_endpoint):
        thresholds = np.concatenate((thresholds, [upper_endpoint]))
    print(f"Intervals endpoints and midpoints for category labels: {thresholds}")
    n = len(thresholds)-1
    if category_labels is None: category_labels = [i for i in range(n)]
    out = {}
    for i, label in enumerate(category_labels): 
        if (i==0) and (thresholds[i]==-np.inf): 
            out[label] = np.logical_and(thresholds[i] < data, data <= thresholds[i+1])
            print(f"{i+1}. {label}: {thresholds[i]} < data <= {thresholds[i+1]}")
        elif (i==n-1) and (thresholds[i]==np.inf): 
            out[label] = np.logical_and(thresholds[i] <= data, data < thresholds[i+1])
            print(f"{i+1}. {label}: {thresholds[i]} <= data < {thresholds[i+1]}")
        else:
            out[label] = np.logical_and(thresholds[i] <= data, data < thresholds[i+1])
            print(f"{i+1}. {label}: {thresholds[i]} <= data < {thresholds[i+1]}")
    if i!=n-1: print(f"There are {n-1-i} categories to be added,{i,n}")
    return out