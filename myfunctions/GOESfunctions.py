import matplotlib.colors as mcolors
def definingColormaps(disp=True):
    # Defining RGB values for RRQPEF colormap
    rgb_values = [
        [255, 255, 255],
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
    # Normalize the RGB values to the range [0, 1]
    colors = [tuple(rgb / 255.0 for rgb in rgb_value) for rgb_value in rgb_values]
    # Create the colormap
    colormaps = {
        'ABI-L2-DSRF':'turbo',
        'ABI-L2-ACMF':'Blues', # Clear Sky Mask
        'ABI-L2-TPWF':'terrain',
        'ABI-L2-LSTF':'jet', # Land Surface Temperature
        'ABI-L2-RRQPEF': mcolors.ListedColormap(colors),
        "ABI-L2-ACHAF": 'ocean', # Cloud Top Height
        "ABI-L2-ACHTF": 'jet', # Cloud Top Temperature
        }
    if disp:
        display(colormaps)
    return colormaps

import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import numpy as np
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

import re
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