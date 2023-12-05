import os, requests
from datetime import datetime
import pandas as pd
import rioxarray as rxr
import scrapy
import scrapydo
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from bs4 import BeautifulSoup
from GOESutils.GlobalVars import *

def QueringGeoColorTif(url = "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/GEOCOLOR/", looking_for=".tif"):
    
    # Send a GET request to the URL to download the page content
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        # Extract text from the <pre> section
        pre_text = soup.find('pre').get_text()
        # Split the text into lines
        lines = pre_text.split('\n')
        # Create a list to hold the data
        data = []
        # Iterate through the lines and split into columns
        n_lines = len(lines)
        for line in lines[:]:
            parts = line.split()
            if len(parts) >= 4:
                filename = parts[-4]
                date = " ".join(parts[-3:-1])
                size = parts[-1]
                data.append([filename, date, int(size)/ (1024 ** 2)])

        # Create a Pandas DataFrame
        df = pd.DataFrame(data, columns=['Filename', 'Date', 'Size'])
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(utc)
        df = df[df["Filename"].str.endswith(looking_for)]
    else:
        print("Failed to download the page. HTTP status code:", response.status_code)
    
    return df


def GeoColorTif(dst_path = destination_path, mode="latest"):
    url = "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/GEOCOLOR/"
    
    if mode == "latest":
        TifFullName = os.path.join(url, "GOES16-ABI-FD-GEOCOLOR-10848x10848.tif")
        data = rxr.open_rasterio(TifFullName)
        data_peru = data.sel(x=slice(PeruLimits_deg[0], PeruLimits_deg[1]), y=slice(PeruLimits_deg[3], PeruLimits_deg[2]))
        old_attrs = data_peru
        class Sha256Spider(scrapy.Spider): # Define the Spider class
            name = 'sha256'
            start_urls = ['https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/GEOCOLOR/GOES16-ABI-FD-GEOCOLOR-10848x10848.tif.sha256']
            def parse(self, response):
                content = response.text
                lines = content.split('\n')
                if len(lines) >= 1:
                    sha256_hash, filename = lines[0].split()
                    yield {
                        'SHA256 Hash': sha256_hash,
                        'Filename': filename
                    }
                else:
                    self.log("Invalid content format")
        scrapydo.setup() # Set up Scrapydo to work with Jupyter
        process = CrawlerProcess(get_project_settings()) # Create a Scrapy process
        content = scrapydo.run_spider(Sha256Spider, process=process) # Run the Spider using Scrapydo in Jupyter Notebook
        TifFileName = content[0]["Filename"]
        ImgTime = datetime.strptime(TifFileName[:11], '%Y%j%H%M')
        img_year, img_month, img_day = str(ImgTime.year), str(ImgTime.month).zfill(2), str(ImgTime.day).zfill(2)
        img_hour, img_minute = str(ImgTime.hour).zfill(2), str(ImgTime.minute).zfill(2)
        identifier = "GeoColor"
        ImageName = '_'.join([identifier, img_year, img_month, img_day, img_hour, img_minute])+'.png'
        ImagePath = os.path.join(dst_path,'Products',identifier)
        FullImageName = os.path.join(ImagePath, ImageName)
        
        
    elif mode == "timerange":
        TifFilesInfo = QueringGeoColorTif(url)
        TifFile = TifFilesInfo.iloc[-2]
        TifFileName = TifFile["Filename"]
        TifFileWebFullPath = os.path.join(url, TifFileName)
        data_tif = rxr.open_rasterio(TifFileWebFullPath)
        data_tif_peru = data_tif.sel(x=slice(PeruLimits_deg[0], PeruLimits_deg[1]), y=slice(PeruLimits_deg[3], PeruLimits_deg[2]))
    
        ImgTime = TifFile["Date"]
        img_year, img_month, img_day = str(ImgTime.year), str(ImgTime.month).zfill(2), str(ImgTime.day).zfill(2)
        img_hour, img_minute = str(ImgTime.hour).zfill(2), str(ImgTime.minute).zfill(2)
        identifier = "GeoColor"
        ImageName = '_'.join([identifier, img_year, img_month, img_day, img_hour, img_minute])+'.png'
        ImagePath = os.path.join(dst_path,'Products',identifier)
        FullImageName = os.path.join(ImagePath, ImageName)
    GeoColorParams = dict(FileName = TifFileName, 
                            ImageTime = ImgTime,
                            ImageTime_str = ImgTime.strftime("%d-%b-%Y %H:%M %Z"),
                            ImageName = ImageName, ImagePath = ImagePath, FullImageName = FullImageName)
    data_peru.attrs["GeoColorParams"] = GeoColorParams
    
    return data_peru