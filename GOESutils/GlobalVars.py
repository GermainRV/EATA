#==================== We set product parameters to download ====================
product_list = { # ABI Products
"ABI-L2-ACHAF": "Cloud Top Height",
"ABI-L2-ACHTF": "Cloud Top Temperature",
"ABI-L2-ACMF": "Clear Sky  Mask",
"ABI-L2-ACTPF": "Cloud Top Phase",
## "ABI-L2-DMWVF": "Derived Motion Winds - Vapor",
## "ABI-L2-DSRF": "Downward Shortwave Radiation",
"ABI-L2-LSTF": "Land Surface Temperature",
"ABI-L2-RRQPEF": "Rainfall rate",
"ABI-L2-TPWF": "Total Precipitable Water",
}
products = list(product_list)

destination_path = '../GOESimages/'
PeruLimits_deg = [-85, -67.5, -20.5, 1.0] # Define the coordinates of the bounding box around Peru
#==================== Setting up time reference variables ====================
from datetime import datetime
import pytz
utc = pytz.timezone('UTC') # UTC timezone
utcm5 = pytz.timezone('America/Lima') # UTC-5 timezone