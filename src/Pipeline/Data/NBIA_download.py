from nbiatoolkit import NBIAClient
import os

def main():
    collection_name = 'Lung-PET-CT-Dx'
    dir_name = "manifest-160866918333/Lung-PET-CT-Dx/"
   
    os.makedirs(dir_name, exist_ok=True)

    with NBIAClient() as client:
        # Get the list of series in the collection
        series_list = client.getSeries(Collection=collection_name)
        
        # Download each series in the collection
        for series in series_list:
            client.downloadSeries(series['SeriesInstanceUID'], dir_name)
            print("Downloaded", series['SeriesInstanceUID'])

    print(f'Collection {collection_name} downloaded to: {dir_name}')

if __name__ == '__main__':
    main()
