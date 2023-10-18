#==================== We set product parameters to download ====================
import goes2go as g2g
destination_path = './GOESimages/'
bucket = 'noaa-goes16'
product_list = { # ABI Products
# "ABI-L2-ACMF": "Clear Sky  Mask",
# "ABI-L2-ACHAF": "Cloud Top Height",
# "ABI-L2-ACTPF": "Cloud Top Phase",
# "ABI-L2-ACHTF": "Cloud Top Temperature",
# "ABI-L2-LSTF": "Land Surface Temperature",
"ABI-L2-RRQPEF": "Rainfall rate",
# "ABI-L2-DSRF": "Downward Shortwave Radiation",
# "ABI-L2-DMWVF": "Derived Motion Winds - Vapor",
# "ABI-L2-TPWF": "Total Precipitable Water",
}
#==================== Setting up time reference variables ====================
from datetime import datetime, timedelta, timezone
import os, pytz, time, requests
utc = pytz.timezone('UTC') # UTC timezone
utcm5 = pytz.timezone('America/Lima') # UTC-5 timezone
initial_date = datetime(2023,9,1)
final_date = datetime.now()
from IPython.display import display, Image, clear_output
# Set the time interval for clearing the output
time_interval = timedelta(minutes=30)
start_time = datetime.now(utcm5)
import GOESutils.GOESplots as gplt
import GOESutils.DataBaseUtils as dbu
import importlib, sys
importlib.reload(sys.modules['GOESutils.GOESplots'])
import GOESutils.GOESplots as gplt

import numpy as np
import matplotlib.pyplot as plt

while True:
    CurrentTime = datetime.now(utcm5)
    CurrentTime_str = CurrentTime.strftime('%Y-%m-%d %H:%M:%S %Z')
    print("============================================================")
    print("Current time is: {}".format(CurrentTime_str))
    RGBdata, GeoColorParams = gplt.GeoColorData(destination_path)
    figGeo, axGeo = gplt.GeoColorPlot(RGBdata, GeoColorParams, toSave=True, toDisplay=False, toUpload=False, dpi=150)

    for product in list(product_list):
        internet = False
        while not internet: # Check for internet connection to download products
            print("Verifying internet connection...")
            try: # In case there is internet connection
                res = requests.get("https://www.google.com/")
                if res.status_code == 200:
                    try: 
                        prodFileList = g2g.data.goes_latest(satellite=bucket, product=product, domain='F', download=True, save_dir=destination_path, return_as='filelist')
                        # prodFileList = g2g.data.goes_nearesttime(datetime.now(utc).replace(tzinfo=None) - timedelta(days=1) + timedelta(hours=5), satellite=bucket, product=product, domain='F', download=True, save_dir=destination_path, return_as='filelist')
                        print("Getting latest available")
                    except ValueError:
                        try:
                            CurrentTime = utc.localize(datetime.now()).replace(tzinfo=None)
                            prodFileList = g2g.data.goes_nearesttime(CurrentTime, satellite=bucket, product=product, domain='F', download=True, save_dir=destination_path, return_as='filelist')
                            print("Getting nearest available")
                        except ValueError:
                            CurrentTime = utc.localize(datetime.now()).replace(tzinfo=None) - timedelta(hours=1)
                            prodFileList = g2g.data.goes_nearesttime(CurrentTime, satellite=bucket, product=product, domain='F', download=True, save_dir=destination_path, return_as='filelist')
                            print("Getting nearest available 1 hour before")
                internet = True
            except Exception as e: # Waiting for internet connection
                print(f"Internet connection lost..{e}")
                internet = False
                print(f"Waiting for internet connection.")
                time.sleep(30)
            time.sleep(1)
        # old_files = os.listdir(ImgPath)
        # old_png_files = set([file for file in old_files if file.endswith('.png')])
        # are_there_old_filenames = (len(old_png_files) > 0)
        
        for f in list(prodFileList['file']): # Reading each file downloaded
            print("Working with file: {}".format(os.path.basename(f)))                
            FullFilePath = os.path.join(destination_path,f)
            data_re, ProductParams = gplt.ProductData(FullFilePath, product, n=4)
            if os.path.exists(ProductParams["FullImagePath"]): # If png image exists, it is shown
                print("Image '{}' already exists in '{}'".format(ProductParams["ImageName"],ProductParams["ImagePath"]))
                display(Image(filename=ProductParams["FullImagePath"], width=640))
            else: # Creating png image
                if (not os.path.exists(ProductParams["ImagePath"])):
                    print(f"Directory for product {product} does not exist. Creating new one...") 
                    os.makedirs(ProductParams["ImagePath"])
                print(f"Image for file {os.path.basename(f)} not found, creating one...")
                figProd = gplt.ProductPlot(data_re, product, axGeo, ProductParams, toSave=True, toDisplay=False, toUpload=False, dpi=150)

        gplt.DepartmentPlot(gplt.departments, product, RGBdata, GeoColorParams, data_re, ProductParams, toSave=True, toDisplay=False, toUpload=False)
            
        print("All the files have been processed.")
        
    # Check if it's time to clear the output
    try:
        CurrentTime = datetime.now(utcm5)
        hour, minute, seconds = CurrentTime.hour, CurrentTime.minute, CurrentTime.second
        elapsed_time = CurrentTime - start_time
        if elapsed_time >= time_interval:
            # Clear the output
            clear_output(wait=True)
            # Reset the start time
            start_time = datetime.now(utcm5)
        if (elapsed_time >= timedelta(hours=3)) or (hour==23 and minute>30):
            for product in list(product_list):
                dbu.eliminar(product.split("-")[-1][:-1])
            
        total_remaining_seconds = (10 - (int(minute) % 10)) * 60 - int(seconds)
        remaining_minutes = total_remaining_seconds // 60
        print("Waiting {} minutes for the next file upload".format(remaining_minutes + 1))
        time.sleep((remaining_minutes+1)*60)
    except Exception as e:
        print("Elapsed time attempting failed.")