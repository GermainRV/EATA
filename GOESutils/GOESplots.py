import matplotlib.colors as mcolors
import colormaps as cmaps
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 8
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from datetime import datetime, timedelta
import pytz, time
import os, re, copy, requests
import goes2go as g2g
from toolbox.wind import spddir_to_uv
from toolbox.cartopy_tools_OLD import common_features, pc
from paint.standard2 import cm_wind
import GOESutils.DataBaseUtils as dbu

#==================== Setting up time reference variables ====================
utc = pytz.timezone('UTC') # UTC timezone
utcm5 = pytz.timezone('America/Lima') # UTC-5 timezone

#==================== Creating georeferenced variables ====================
map_proj_pc = ccrs.PlateCarree(), "PlateCarree projection"
# Add coastlines feature
# coastlines_feature = cfeature.NaturalEarthFeature(
#     category='physical',
#     name='coastline',
#     scale='10m',
#     edgecolor='black',
#     facecolor='none')
# # Add country boundaries feature
# countries_feature = cfeature.NaturalEarthFeature(
#     category='cultural',
#     name='admin_0_countries',
#     scale='10m',
#     edgecolor='black',
#     facecolor='none')
# Create the polygon representing the bounding box
PeruLimits_deg = [-85, -67.5, -20.5, 1.0] # Define the coordinates of the bounding box around Peru
peru_box = Polygon([(PeruLimits_deg[0], PeruLimits_deg[2]), (PeruLimits_deg[1], PeruLimits_deg[2]), (PeruLimits_deg[1], PeruLimits_deg[3]), (PeruLimits_deg[0], PeruLimits_deg[3])])
# gdf_coastline = gpd.read_file("./Boundaries/ne_10m_coastline/ne_10m_coastline.shp", mask=peru_box)
gdf_maritime = gpd.read_file("./Boundaries/World_EEZ_v11_20191118/eez_v11.shp", mask=peru_box)
gdf_countries = gpd.read_file("./Boundaries/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp", mask=peru_box)
gdf_states = gpd.read_file("./Boundaries/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp", mask=peru_box)
gdf_peru_land = gpd.read_file("./Boundaries/PER_adm/PER_adm2.shp")
# Filter the GeoDataFrame to keep only rows where adm1_code matches "PER"
gdf_peru_sea = gdf_maritime[gdf_maritime["TERRITORY1"] == "Peru"].iloc[[1]]
gdf_countries = gdf_countries[gdf_countries["ADMIN"] != "Peru"]
gdf_states = gdf_states[gdf_states["adm1_code"].str[:3] == "PER"]
pol_lima = gdf_states[gdf_states['name'] == "Lima"].geometry.iloc[0]
pol_lima_prov = gdf_states[gdf_states['name'] == "Lima Province"].geometry.iloc[0]
pol_callao = gdf_states[gdf_states['name'] == "Callao"].geometry.iloc[0]
ind_lima = gdf_states[gdf_states['name'] == "Lima"].index[0]
gdf_states.at[ind_lima,'geometry'] = pol_lima.union(pol_callao)#.union(pol_lima_prov)
gdf_states = gdf_states[~gdf_states['name'].isin(["Callao"])]#"Lima Province", 

pol_lima = gdf_peru_land[gdf_peru_land['NAME_1'] == "Lima"].geometry.iloc[0]
pol_lima_prov = gdf_peru_land[gdf_peru_land['NAME_1'] == "Lima Province"].geometry.iloc[0]
pol_callao = gdf_peru_land[gdf_peru_land['NAME_1'] == "Callao"].geometry.iloc[0]
ind_lima = gdf_peru_land[gdf_peru_land['NAME_1'] == "Lima"].index[0]
gdf_peru_land.at[ind_lima,'geometry'] = pol_lima.union(pol_callao)#.union(pol_lima_prov)
gdf_peru_land = gdf_peru_land[~gdf_peru_land['NAME_1'].isin(["Callao"])]#"Lima Province", 
departments = gdf_peru_land["NAME_1"].unique().tolist()

def definingColormaps(disp=True):
    # Defining RGB values for RRQPEF colormap
    rgb_values = [
        [127, 127, 127],
        [0, 200, 255],
        [0, 163, 255],
        [0, 82, 255],
        [0, 0, 200],
        [150, 255, 150],
        [50, 200, 50],
        [0, 130, 0],
        [255, 255, 0],
        [170, 170, 0],
        [255, 127, 0],
        [200, 70, 70],
        [255, 160, 160],
        [255, 0, 0],
        [157, 0, 157],
        [0, 0, 0],
        [222, 222, 222]]
    rgb_values = [[0, 0, 250], [0, 250, 250], [67, 80, 126], [205,240,254], [120,120,120]]
    # Normalize the RGB values to the range [0, 1]
    colors = [tuple(rgb / 255.0 for rgb in rgb_value) for rgb_value in rgb_values]
    # Create the colormap
    # RRQPEcmap = mcolors.ListedColormap(colors)
    # RRQPEcmap.set_bad('w', alpha=0)
    ACTPcmap = mcolors.ListedColormap(colors)
    product_colormaps = {
        'ABI-L2-DSRF':'turbo',
        'ABI-L2-ACMF': dict(cmap=cmaps.greys_light, min = 1, max = 4), # Clear Sky Mask
        'ABI-L2-TPWF': dict(cmap='Greens', min = 0, max = 80),
        'ABI-L2-LSTF': dict(cmap='jet', min = 0, max = 50), # Land Surface Temperature
        'ABI-L2-RRQPEF': dict(cmap=cmaps.deep, min = 0, max = 100),#, cmaps.ncview_default
        "ABI-L2-ACHAF": dict(cmap=cmaps.GMT_drywet, min = 0, max = 17), # Cloud Top Height
        "ABI-L2-ACHTF": dict(cmap='jet', min = -100, max = 50), # Cloud Top Temperature
        'ABI-L2-ACTPF': dict(cmap=ACTPcmap, min = 1, max = 6), # cmaps.cosmic_r
        'ABI-L2-DMWVF': dict(cmap='jet'),
        }
    if disp:
        display(product_colormaps)
    return product_colormaps

def get_image_params(data, identifier, satellite='goes16', destination_path='./GOESimages/'):
    ImgTitle = data.attrs['title'].split("ABI L2 ")[-1]
    ImgTitle = ImgTitle.split(" - ")[0]
    format_string = '%Y-%m-%dT%H:%M:%S.%fZ'
    time_coverage_start = datetime.strptime(data.attrs['time_coverage_start'], format_string)
    time_coverage_end = datetime.strptime(data.attrs['time_coverage_end'], format_string)
    ImgTime = time_coverage_end
    # ImgTime = time_coverage_start + (time_coverage_end - time_coverage_start) / 2
    ImgTime = ImgTime.replace(tzinfo = utc)
    ImgTime = ImgTime.astimezone(utcm5)
    ImgTime_str = ImgTime.strftime('%H:%M UTC%Z %d-%m-%Y') # '%H:%M UTC %d-%b-%Y'
    varnames = list(data.data_vars.keys())
    spatial_res = data.attrs["spatial_resolution"].split()[0]
    # spatial_res = float(re.findall('\d+',spatial_res)[0])
    # Building image name format
    img_year, img_month, img_day = str(ImgTime.year), str(ImgTime.month).zfill(2), str(ImgTime.day).zfill(2)
    img_hour, img_minute = str(ImgTime.hour).zfill(2), str(ImgTime.minute).zfill(2)
    ImageName = '_'.join([satellite, identifier, img_year, img_month, img_day, img_hour, img_minute])+'.png'
    ImagePath = os.path.join(destination_path,'Products',identifier)
    FullImagePath = os.path.join(ImagePath,ImageName)
    out = {
            'FileName': data.attrs["dataset_name"],
            'ImageTitle': ImgTitle,
            'ImageTime':ImgTime, 'ImageTime_str':ImgTime_str,
            'VarNames':varnames, 'SpatialResolution': spatial_res,
            'ImageName': ImageName, 'ImagePath': ImagePath, 'FullImagePath': FullImagePath,
            'DataAttrs': data.attrs}
    return out

def GeoColorData(destination_path, mode="latest", file_datetime=datetime.now()):
    file_datetime = file_datetime.astimezone(utc).replace(tzinfo=None)
    internet = False
    while not internet:
        print("Verifying internet connection...")
        try: # Check for internet connection
            res = requests.get("https://www.google.com/")
            if res.status_code == 200:
                if(mode=="latest"):
                    try: 
                        gFileList = g2g.data.goes_latest(satellite='noaa-goes16', product='ABI-L2-MCMIP', domain='F', download=True, save_dir=destination_path, return_as='filelist')
                        print("Getting latest available")
                    except:
                        try:
                            print("Latest file available probably failed to download, rewriting existing...")
                            gFileList = g2g.data.goes_latest(satellite='noaa-goes16', product='ABI-L2-MCMIP', domain='F', download=True, save_dir=destination_path, return_as='filelist', overwrite=False)
                        except ValueError:
                            try:
                                current_datetime = datetime.utcnow()
                                gFileList = g2g.data.goes_nearesttime(current_datetime, satellite='noaa-goes16', product='ABI-L2-MCMIP', domain='F', download=True, save_dir=destination_path, return_as='filelist')
                                print("Getting nearest available")
                            except ValueError:
                                current_datetime = datetime.utcnow() - timedelta(hours=1)
                                gFileList = g2g.data.goes_nearesttime(current_datetime, satellite='noaa-goes16', product='ABI-L2-MCMIP', domain='F', download=True, save_dir=destination_path, return_as='filelist')
                                print("Getting nearest available 1 hour before")
                elif(mode=="nearesttime"):
                    gFileList = g2g.data.goes_nearesttime(file_datetime,satellite='noaa-goes16', product='ABI-L2-MCMIP', domain='F', download=True, save_dir=destination_path, return_as='filelist')
            internet = True
        except Exception as e: # Waiting for internet connection
            print(f"Internet connection lost..{e}")
            internet = False
            print(f"Waiting for internet connection.")
            time.sleep(30)
        time.sleep(1)
    
    gdata = xr.open_dataset(os.path.join(destination_path,gFileList['file'][0]), engine='rasterio').isel(band=0)
    crs_obj = gdata.rio.crs
    crs_dest = map_proj_pc[0] # "EPSG:4326"
    GeoColorParams = get_image_params(gdata, identifier="GeoColor")
    print(f"Reading file {GeoColorParams['FileName']} as geocolor image.")
    ImageTime = GeoColorParams['ImageTime']
    isDay = (ImageTime.hour>5 and ImageTime.hour<17)
    if isDay: GeoColor = gdata.rgb.NaturalColor(night_IR=False).rio.write_crs(crs_obj)
    else: GeoColor = gdata.rgb.TrueColor(night_IR=True).rio.write_crs(crs_obj)
        
    R, G, B = [
        GeoColor.isel(rgb=i).rio.reproject(crs_dest) # , resolution=0.01
        .sel(x=slice(PeruLimits_deg[0], PeruLimits_deg[1]), y=slice(PeruLimits_deg[3], PeruLimits_deg[2]))
        for i in range(3)
    ]
    RGBdata = xr.concat([R, G, B], dim='rgb').transpose('y', 'x', 'rgb')
    return RGBdata, GeoColorParams
    
def GeoColorPlot(RGBdata, GeoColorParams, toSave=False, toDisplay=False, toUpload=False, department=False, dpi=300):
    ImageTime = GeoColorParams["ImageTime"]
    print(f"Plotting geocolor image at {ImageTime}.")
    ImageTime = GeoColorParams['ImageTime']
    isDay = (ImageTime.hour>5 and ImageTime.hour<17)
    if isDay: 
        print("It is daytime! Plotting NaturalColor image...")
        edgecolor, gridcolor = 'white', 'black'
    else: 
        print("It is nighttime! Plotting TrueColor image...")
        edgecolor, gridcolor = 'white', 'darkgray'
    x_coords, y_coords = RGBdata.x.values, RGBdata.y.values
    fig, ax = plt.subplots(figsize=(8, 12), subplot_kw=dict(projection=map_proj_pc[0]))
    # ax.set_extent(PeruLimits_deg)
    ax.imshow(RGBdata, extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()), origin='upper')
    # ax.add_feature(coastlines_feature, linewidth=0.75, edgecolor=edgecolor)
    # ax.add_feature(countries_feature, linewidth=0.75, edgecolor=edgecolor)
    ax.add_geometries(gdf_countries['geometry'], crs=map_proj_pc[0], facecolor='none', edgecolor=edgecolor, linewidth=0.75)
    ax.add_geometries(gdf_states['geometry'], crs=map_proj_pc[0], facecolor='none', edgecolor=edgecolor, linewidth=0.75)
    if not department: ax.add_geometries(gdf_peru_sea['geometry'], crs=map_proj_pc[0], facecolor='none', edgecolor=edgecolor, linewidth=0.75)
    # ax.gridlines(draw_labels=True, lw=0.75, color=gridcolor, alpha=0.7, ls='--')
    # ax.set_title(title)
    if toSave:
        if not os.path.exists(GeoColorParams["ImagePath"]): os.makedirs(GeoColorParams["ImagePath"])
        fig.savefig(GeoColorParams["FullImagePath"],dpi=dpi,bbox_inches='tight')
        
    if toDisplay: plt.show()
    else: plt.close()
    
    if toUpload:
        database_folder = os.path.join("GOESimages", product.split("-")[-1][:-1])
        dbu.subir(GeoColorParams["FullImagePath"], database_folder, GeoColorParams["ImageName"])
    
    return fig, ax

def ProductData(FullFilePath, product, bucket='noaa-goes16', n=1):
    """
    Extracts and processes specific product data from a GOES satellite dataset.

    Inputs:
        data (xarray.Dataset): GOES satellite dataset.
        product (str): The product identifier.

    Returns:
        data_re (xarray.DataArray): Processed data for the specified product.
        ProductParams (dict): Product parameters.
    """
    isACM = isACHA = isACTP = isACHT = isLST = isRRQPE = isDSR = isDMWV = isTPW = False
    if (product=="ABI-L2-ACMF"): isACM = True
    elif (product=="ABI-L2-ACHAF"): isACHA = True
    elif (product=="ABI-L2-ACTPF"): isACTP = True
    elif (product=="ABI-L2-ACHTF"): isACHT = True
    elif (product=="ABI-L2-LSTF"): isLST = True
    elif (product=="ABI-L2-RRQPEF"): isRRQPE = True 
    elif (product=="ABI-L2-DSRF"): isDSR = True
    elif (product=="ABI-L2-DMWVF"): isDMWV = True
    elif (product=="ABI-L2-TPWF"): isTPW = True
    identifier = product.split("-")[-1][:-1]
    
    if isDMWV:
        data_re = xr.open_dataset(FullFilePath, engine='netcdf4')
        ProductParams = get_image_params(data_re, identifier)
    else:
        data = xr.open_dataset(FullFilePath, engine='rasterio')
        crs_obj = data.rio.crs
        ProductParams = get_image_params(data, identifier)
        if isACHA or isACTP or isACHT or isLST or isRRQPE or isDSR or isTPW: varname = ProductParams["VarNames"][0]
        elif isACM: varname = ProductParams["VarNames"][1]
        
        data = data.isel(band=0)[varname]
        data_re = data.rio.reproject(map_proj_pc[0]) # , resolution=0.01
        data_re = data_re.sel(x=slice(PeruLimits_deg[0], PeruLimits_deg[1]), y=slice(PeruLimits_deg[3], PeruLimits_deg[2]))
        data_re = data_re.isel(y=slice(None, None, -1))
        
        if n==0 or n==1:
            print("No interpolation performed")
        else:
            # Interpolation
            x, y = data_re.x.values, data_re.y.values
            xnew, ynew = np.linspace(x[0], x[-1], num=n*len(x)), np.linspace(y[0], y[-1], num=n*len(y))
            if isACHT or isLST: 
                if isLST:
                    for i in range(20):
                        data_re = data_re.interpolate_na(dim="y", method="linear", limit=1, use_coordinate=True)
                        data_re = data_re.interpolate_na(dim="x", method="linear", limit=1, use_coordinate=True)
                else: 
                    data_re = data_re.interpolate_na(dim="x", method="linear", limit=2)
                    data_re = data_re.interpolate_na(dim="y", method="linear", limit=2)
                data_re = data_re.interp(x=xnew, y=ynew)
            else: 
                data_re = data_re.fillna(0).interp(x=xnew, y=ynew)
                data_re = data_re.where(data_re >= 0, other=0)
        
        # Processing
        attributes = data_re.attrs 
        if isACM:
            # mask0 = (data_re.values < 0.5)
            # mask1 = (data_re.values >= 0.5) | (data_re.values < 1.5)
            # mask2 = (data_re.values >= 1.5) | (data_re.values < 2.5)
            # mask3 = (data_re.values >= 2.5)
            # data_re.values[mask0] = 0
            # data_re.values[mask1] = 1
            # data_re.values[mask2] = 2
            # data_re.values[mask3] = 3
            mask = (data_re.values == 0)
            # data_re.values[mask] = np.nan
        elif isACHT or isLST:
            data_re = data_re - 273.15
            data_re.attrs = attributes
            if data_re.units=="K": data_re.attrs["units"] = "°C"
        else: # isACTP or isACHA or isRRQPE
            if isACHA:
                data_re = data_re/1e3
                data_re.attrs = attributes
                if data_re.units=="m": data_re.attrs["units"] = "km"
            mask = (data_re.values == 0)
            data_re.values[mask] = np.nan
    # data_re = data_re.rio.write_crs(map_proj_pc[0])
    return data_re, ProductParams

def ProductPlot(data_re, product, axGeo, ProductParams, toSave=False, toDisplay=False, toUpload=False, title="Peru", dpi=300):
    """
    Plots a processed product data from a GOES satellite dataset.

    Inputs:
        data_re (xarray.DataArray): Processed data for the specified product.
        product (str): The product identifier.
        axGeo (matplotlib.axes._subplots.GeoAxesSubplot): Matplotlib axis for plotting.
        ProductParams (dict): Product parameters.
        toSave (bool, optional): Whether to save the plot as an image file. Default is False.

    Returns:
        figProd (matplotlib.figure.Figure): The figure containing the plot.
    """
    isACM = isACHA = isACTP = isACHT = isLST = isRRQPE = isDMWV = isTPW = False
    if (product=="ABI-L2-ACMF"): isACM = True
    elif (product=="ABI-L2-ACHAF"): isACHA = True
    elif (product=="ABI-L2-ACTPF"): isACTP = True
    elif (product=="ABI-L2-ACHTF"): isACHT = True
    elif (product=="ABI-L2-LSTF"): isLST = True
    elif (product=="ABI-L2-RRQPEF"): isRRQPE = True 
    elif (product=="ABI-L2-DMWVF"): isDMWV = True
    elif (product=="ABI-L2-TPWF"): isTPW = True
    
    prod_cmap_dic = definingColormaps(False)[product]
    product_cmap = prod_cmap_dic["cmap"]
    
    axProd = copy.deepcopy(axGeo)
    cbar_fontsize = 10
    if isACM or isACTP:
        flag_values = data_re.flag_values#[1:]
        dflag = np.mean(np.diff(flag_values))/2
        flag_meanings = data_re.flag_meanings.split()#[1:]
        flag_meanings = [flag.replace("_", "\n") for flag in flag_meanings]
        if isACM:
            nbin = len(flag_values)
            product_cmap = product_cmap.discrete(nbin)
        im = axProd.pcolormesh(data_re.x, data_re.y, data_re.values, cmap=product_cmap, vmin=flag_values[0] - dflag, vmax=flag_values[-1] + dflag)
        cbar = plt.colorbar(im,ax=axProd, orientation='horizontal',
                            shrink=0.7, pad=0.01)
        cbar.set_ticks(flag_values)
        cbar.set_ticklabels(flag_meanings)
        # units_latex = re.sub(r'(\w)(-)(\d)', r'\1^{-\3}', data_re.units)
        cbar.set_label(r"{}".format(data_re.long_name), size=cbar_fontsize)
    elif isACHA or isACHT or isLST or isRRQPE or isTPW:
        im = axProd.pcolormesh(data_re.x, data_re.y, data_re.values, cmap=product_cmap, vmin=prod_cmap_dic["min"], vmax=prod_cmap_dic["max"])
        if isRRQPE: cbar_extent = "max"
        else: cbar_extent = "both"
        cbar = plt.colorbar(im,ax=axProd, orientation='horizontal', extend=cbar_extent, shrink=0.7, pad=0.01)
        units_latex = re.sub(r'(\w)(-)(\d)', r'\1^{-\3}', data_re.units)
        cbar.set_label(r"{} $({})$".format(data_re.long_name,units_latex), size=cbar_fontsize)
    elif isDMWV:
        # Convert GOES wind speed and direction to u- and v-wind components
        gu, gv = spddir_to_uv(data_re.wind_speed, data_re.wind_direction)
        im = axProd.quiver(
            data_re.lon.data,
            data_re.lat.data,
            gu.data,
            gv.data,
            data_re.wind_speed,
            **cm_wind().cmap_kwargs,
            scale=30, scale_units='xy', angles='xy',
            transform=pc
        )
        # axProd.gridlines(draw_labels=False, lw=0.75, color='darkgray', alpha=0.7, ls='--')
        cbar = plt.colorbar(im,ax=axProd, orientation='horizontal', shrink=1, pad=0.01)
        units_latex = re.sub(r'(\w)(-)(\d)', r'\1^{-\3}', data_re.wind_speed.units)
        cbar.set_label(r"{} $({})$".format(data_re.wind_speed.long_name,units_latex), size=cbar_fontsize)
    # cbar.set_label(label="asa",size=12)
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    axProd.set_title(f"{ProductParams['ImageTime_str']}\n{title}", loc="right")
    axProd.set_title(f"{ProductParams['ImageTitle']}", loc='left', fontweight='bold', fontsize=10)
    # axProd.set_title("Peru image from satellite GOES\n {}".format(ProductParams['ImgTime_str']))
    
    if toSave:
        if not os.path.exists(ProductParams["ImagePath"]): os.makedirs(ProductParams["ImagePath"])
        axProd.figure.savefig(ProductParams["FullImagePath"], dpi=dpi, bbox_inches='tight')
        print("Image {} saved in '{}'".format(ProductParams["ImageName"], ProductParams["ImagePath"]))
    plt.close()
    figProd = axProd.figure
    
    if toDisplay: display(figProd)
    
    if toUpload:
        database_folder = os.path.join("GOESimages", product.split("-")[-1][:-1], "Peru")
        dbu.subir(ProductParams["FullImagePath"], database_folder, ProductParams["ImageName"])
    
    return figProd

def DepartmentPlot(departments, product, RGBdata, GeoColorParams, data_re, ProductParams, toSave=False, toDisplay=False, toUpload=False, dpi=300):
    for dep in departments: # Getting georreferenced product by each department
        bounding_box = gdf_peru_land[gdf_peru_land["NAME_1"] == dep].geometry.bounds.agg({"minx": "min", "miny": "min", "maxx": "max", "maxy": "max"})
        bounding_box = np.array([bounding_box.minx, bounding_box.maxx, bounding_box.miny, bounding_box.maxy])
        figGeo_dep, axGeo_dep = GeoColorPlot(RGBdata.sel(x=slice(bounding_box[0], bounding_box[1]), y=slice(bounding_box[3], bounding_box[2])), GeoColorParams, department=True, toDisplay=False)
        data_dep = data_re.sel(x=slice(bounding_box[0], bounding_box[1]), y=slice(bounding_box[2], bounding_box[3]))
        figProd_dep = ProductPlot(data_dep, product, axGeo_dep, ProductParams, title=dep+", Peru")
        if toSave:
            DepImagePath = os.path.join(ProductParams["ImagePath"], dep.replace(" ","_"))
            if not os.path.exists(DepImagePath): os.makedirs(DepImagePath)
            DepImageName = ProductParams["ImageName"].split(".")[0] + "_" + dep + ".png"
            DepFullImagePath = os.path.join(DepImagePath, DepImageName)
            figProd_dep.savefig(DepFullImagePath, dpi=dpi, bbox_inches='tight')
            print("Image {} saved in '{}'".format(DepImageName, DepImagePath))
            
        if toDisplay: display(figProd_dep)
        
        if toSave and toUpload:
            database_folder = os.path.join("GOESimages", product.split("-")[-1][:-1], dep)
            dbu.subir(DepFullImagePath, database_folder, DepImageName)

def plotBothProjections(data,global_variables):
    variable_names = ['data','imgExtention', 'coords', 'map_proj_src','varname','product_cmap',
                      'coastlines_feature','countries_feature','map_proj_dst']
    # for var in variable_names:
    #     # exec(var+" = global_variables.get('"+var+"')")
    #     print(var+" = global_variables.get('"+var+"')")
    data = global_variables.get('data')
    imgExtention = global_variables.get('imgExtention')
    coords = global_variables.get('coords')
    map_proj_src = global_variables.get('map_proj_src')
    varname = global_variables.get('varname')
    product_cmap = global_variables.get('product_cmap')
    coastlines_feature = global_variables.get('coastlines_feature')
    countries_feature = global_variables.get('countries_feature')
    map_proj_dst = global_variables.get('map_proj_dst')
    extent_deg = np.copy(imgExtention)
    if(coords == "xy"):
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(1, 2, 1, projection=map_proj_src[0])
        extent_deg = ax1.get_extent()
        im1 = ax1.imshow(data[varname].values, transform=map_proj_src[0], extent=extent_deg, origin='upper', cmap=product_cmap)
        ax1.add_feature(coastlines_feature, linewidth=0.75)
        ax1.add_feature(countries_feature, linewidth=0.75)
        ax1.gridlines(draw_labels=True,lw=0.75,color='k',alpha=0.75,ls='--')
        ax1.set_title("Original image: "+map_proj_src[1],verticalalignment='bottom')

        ax2 = fig.add_subplot(1, 2, 2, projection=map_proj_dst[0])
        ax2.set_extent(imgExtention) # ax.set_global(), imgExtention, PeruLimits_deg
        extent_deg = ax2.get_extent()
        im2 = ax2.imshow(data[varname].values, transform=map_proj_src[0], origin='upper', cmap=product_cmap)
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.COASTLINE)
        ax2.gridlines(draw_labels=True,lw=0.75,color='k',alpha=0.75,ls='--')
        ax2.set_title("Transformed image: "+map_proj_dst[1])
        plt.show()
        plane_projection_data = im2.get_array().data
    elif(coords == "lonlat"):
        lon, lat = data.lon.values, data.lat.values
        fig = plt.figure(figsize=(10, 8))
        
        ax1 = fig.add_subplot(1, 2, 1, projection=map_proj_src[0])
        im1 = ax1.pcolormesh(lon,lat,data[varname].values,cmap=product_cmap)
        ax1.add_feature(coastlines_feature, linewidth=0.75)
        ax1.add_feature(countries_feature, linewidth=0.75)
        ax1.gridlines(draw_labels=True,lw=0.75,color='k',alpha=0.75,ls='--')
        ax1.set_title("Original image: "+map_proj_src[1])

        ax2 = fig.add_subplot(1, 2, 2, projection=map_proj_dst[0])
        im2 = ax2.pcolormesh(lon,lat,data[varname].values, transform=map_proj_src[0],cmap=product_cmap)
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.COASTLINE)
        ax2.gridlines(draw_labels=True,lw=0.75,color='k',alpha=0.75,ls='--')
        ax2.set_title("Transformed image: "+map_proj_dst[1])
        plt.show()
        plane_projection_data = data[varname].values
    return plane_projection_data, extent_deg

def plotSatImg(data,global_variables):
    variable_names = ['data','imgExtention', 'coords', 'map_proj_src','varname','product_cmap',
                      'coastlines_feature','countries_feature','map_proj_dst']
    # for var in variable_names:
    #     # exec(var+" = global_variables.get('"+var+"')")
    #     print(var+" = global_variables.get('"+var+"')")
    data = global_variables.get('data')
    imgExtention = global_variables.get('imgExtention')
    coords = global_variables.get('coords')
    map_proj_src = global_variables.get('map_proj_src')
    varname = global_variables.get('varname')
    product_cmap = global_variables.get('product_cmap')
    coastlines_feature = global_variables.get('coastlines_feature')
    countries_feature = global_variables.get('countries_feature')
    map_proj_dst = global_variables.get('map_proj_dst')
    selected_product = global_variables.get('selected_product')
    gdf_peru_land = global_variables.get('gdf_peru_land')
    gdf_peru_sea = global_variables.get('gdf_peru_sea')
    str_ImgTime = global_variables.get('str_ImgTime')
    map_proj_pc = global_variables.get('map_proj_pc')
    satellite = global_variables.get('satellite')
    year = global_variables.get('year')
    month = global_variables.get('month')
    day = global_variables.get('day')
    hour = global_variables.get('hour')
    minute = global_variables.get('minute')
    FilePath = global_variables.get('FilePath')
    PeruLimits_deg = global_variables.get('PeruLimits_deg')
    
    if(coords == "xy"):
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection=map_proj_dst[0]))
        ax.set_extent(PeruLimits_deg)
        # im = ax.imshow(transformed_data, origin='lower', transform=map_proj_dst[0], extent=extent_deg, cmap='turbo')
        im = ax.imshow(data[varname].values, transform=map_proj_src[0], cmap=product_cmap)
        cbar = plt.colorbar(im,ax=ax, orientation='horizontal', shrink=0.5, pad=0.05)
        units_latex = re.sub(r'(\w)(-)(\d)', r'\1^{-\3}', data[varname].units)
        if ( selected_product[:-1] == "ABI-L1b-Rad") or (selected_product[:-1] == "ABI-L2-CMIP"):
            cbar.set_label(r"{} $({})$, band={}".format(data.title,units_latex,selected_channel))
        else:
            cbar.set_label(r"{} $({})$".format(data.title,units_latex))
        ax.add_feature(coastlines_feature, linewidth=0.75)
        ax.add_feature(countries_feature, linewidth=0.75)
        ax.add_geometries(gdf_peru_land['geometry'], crs=map_proj_pc[0], facecolor='none', edgecolor='black', linewidth=0.75)
        ax.add_geometries(gdf_peru_sea['geometry'], crs=map_proj_pc[0], facecolor='none', edgecolor='black', linewidth=0.75)
        ax.gridlines(draw_labels=True,lw=0.75,color='k',alpha=0.7,ls='--')
        ax.set_title("GOES Image, Platform: {}, Geographic coverage: {}\n {}".format(data.platform_ID,data.scene_id,str_ImgTime))
        plt.show()
    elif(coords == "lonlat"):
        lon, lat = data.lon.values, data.lat.values
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection=map_proj_src[0]))
        im = ax.pcolormesh(lon,lat,data[varname].values,cmap=product_cmap,transform=map_proj_src[0])
        cbar = plt.colorbar(im,ax=ax, orientation='horizontal', shrink=0.5, pad=0.05)
        units_latex = re.sub(r'(\w)(-)(\d)', r'\1^{-\3}', data[varname].units)
        if ( selected_product[:-1] == "ABI-L1b-Rad") or (selected_product[:-1] == "ABI-L2-CMIP"):
            cbar.set_label(r"{} $({})$, band={}".format(data.title,units_latex,selected_channel))
        else:
            cbar.set_label(r"{} $({})$".format(data.title,units_latex))
        ax.set_extent(PeruLimits_deg)
        ax.add_feature(coastlines_feature, linewidth=0.75)
        ax.add_feature(countries_feature, linewidth=0.75)
        ax.add_geometries(gdf_peru_land['geometry'], crs=map_proj_pc[0], facecolor='none', edgecolor='black', linewidth=0.75)
        ax.add_geometries(gdf_peru_sea['geometry'], crs=map_proj_pc[0], facecolor='none', edgecolor='black', linewidth=0.75)
        ax.gridlines(draw_labels=True,lw=0.75,color='k',alpha=0.7,ls='--')
        ax.set_title("GOES Image, Platform: {}, Geographic coverage: {}\n {}".format(data.platform_ID,data.scene_id,str_ImgTime))
        plt.show()
    ImageName = satellite +'_'+ year +'_'+ month +'_'+ day +'_'+ selected_product.split('-')[-1] +'_'+ hour +'_'+ minute + '.png'
    # plt.savefig(os.path.join(FilePath, ImageName),dpi=300,bbox_inches='tight')
    print("Image '{}' saved in '{}'".format(ImageName,FilePath))
    return


