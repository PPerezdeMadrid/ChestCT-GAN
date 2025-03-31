from nbiatoolkit import NBIAClient
import os

with NBIAClient() as client:
    collection_name = 'Lung-PET-CT-Dx'
    dir_name = "manifest-160866918333/Lung-PET-CT-Dx/"
   
    os.makedirs(dir_name, exist_ok=True)

    # Obtén la lista de series en la colección
    series_list = client.getSeries(Collection=collection_name)
    
    # Descarga cada serie en la colección
    for series in series_list:
        client.downloadSeries(series['SeriesInstanceUID'], dir_name)
        print("Descargado", series['SeriesInstanceUID'])

    print(f'Colección {collection_name} descargada en: {dir_name}')
