import json,os,requests,zipfile


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


if __name__ == "__main__":
    print('Preparing dataset.')
    if not os.path.exists("./java"):
        print('Downloading Java dataset...')
        file_id = '1hVJaA2JA377Iz3bstHLIGaffUh_ogVnG'
        destination = './java.zip'
        download_file_from_google_drive(file_id, destination)
        print('Unzipping Java dataset...')
        fz = zipfile.ZipFile("./java.zip", 'r')
        for file in fz.namelist(): fz.extract(file, "./java/") 
        # print('Clearing zip file...')    
        # os.remove("./data/java.zip")
    if not os.path.exists("./python"):
        print('Downloading Python dataset...')
        file_id = '1lQhczrERskISdBcWeS6VWLwCMpBAh-YF'
        destination = './python.zip'
        download_file_from_google_drive(file_id, destination)
        print('Unzipping Python dataset...')
        fz = zipfile.ZipFile("./python.zip", 'r')
        for file in fz.namelist(): fz.extract(file, "./python/")
        # print('Clearing zip file...')     
        # os.remove("./data/python.zip")
    print('Preprocessed dataset prepared.')