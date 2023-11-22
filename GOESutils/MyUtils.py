import numpy as np
import requests


def check_internet_connection():
    try:
        # Send a GET request to a reliable website (e.g., google.com)
        response = requests.get("https://www.google.com", timeout=5)
        # Check the response status code
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.ConnectionError:
        return False

def interval_categorizer(data, thresholds, category_labels=None, lower_endpoint=-np.inf, upper_endpoint=np.inf, toPrint=False):
    if isinstance(thresholds, list): thresholds = np.array(thresholds)
    if (lower_endpoint == -np.inf) and (thresholds[0] != lower_endpoint): 
        thresholds = np.concatenate(([lower_endpoint], thresholds))
    if (upper_endpoint == np.inf) and (thresholds[-1] != upper_endpoint):
        thresholds = np.concatenate((thresholds, [upper_endpoint]))
    if toPrint: print(f"Intervals endpoints and midpoints for category labels: {thresholds}")
    n = len(thresholds)-1
    if category_labels is None: category_labels = [i for i in range(n)]
    out = {}
    intervals = []
    for i, label in enumerate(category_labels): 
        if (i==0) and (thresholds[i]==-np.inf): 
            out[label] = np.logical_and(thresholds[i] < data, data <= thresholds[i+1])
            if toPrint: print(f"{i+1}. {label}: {thresholds[i]} < data <= {thresholds[i+1]}")
        elif (i==n-1) and (thresholds[i]==np.inf): 
            out[label] = np.logical_and(thresholds[i] <= data, data < thresholds[i+1])
            if toPrint: print(f"{i+1}. {label}: {thresholds[i]} <= data < {thresholds[i+1]}")
        else:
            out[label] = np.logical_and(thresholds[i] <= data, data < thresholds[i+1])
            if toPrint: print(f"{i+1}. {label}: {thresholds[i]} <= data < {thresholds[i+1]}")
        intervals.append([thresholds[i], thresholds[i+1]])
    if i!=n-1: print(f"There are {n-1-i} categories to be added,{i,n}")
    return out, intervals

def format_value(value, base_unit='', scale=1000):
    prefixes = {-3:'n', -2:'u', -1:'m', 0:'', 1:'K', 2:'M', 3:'G', 4:'T'}
    scales = list(prefixes)
    if value == 0:
        unit = prefixes[0]
    else:
        value_scale = int(np.log(np.abs(value))/np.log(scale))
        if not value_scale in scales:
            if value_scale < scales[0]:
                value_scale = scales[0]
            if value_scale > scales[0]:
                value_scale = scales[-1]
        unit = prefixes[value_scale]
        value /= scale**value_scale
    return value, unit+base_unit
    
def print_progress_bar(iteration, total, bar_length=50):
    progress = (iteration / total)
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    percent = int(progress * 100)
    print(f"\rDownloading [{arrow}{spaces}] {percent}% Complete", end='', flush=True)


def generate_hourly_dates(start_date, final_date):
    current_date = start_date
    hourly_resolution = timedelta(hours=1)
    while current_date <= final_date:
        yield current_date
        current_date += hourly_resolution
