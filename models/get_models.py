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
    print('Downloading trained models.')
    if not os.path.exists("./gta_java.zip"):
        print('Downloading Java trained model...')
        file_id = '1vnIuGLBNGU_AHDwL7yZIkoaByWiLKYxb'
        destination = './gta_java.zip'
        download_file_from_google_drive(file_id, destination)
        print('Unzipping Java trained model...')
        fz = zipfile.ZipFile("./gta_java.zip", 'r')
        for file in fz.namelist(): fz.extract(file, "./") 
        # print('Clearing zip file...')    
        # os.remove("./data/gta_java.zip")
    if not os.path.exists("./gta_python"):
        print('Downloading Python trained model...')
        file_id = '1tk3Wc4YpSo_oLKCi6h3Kitvsux3vWFUO'
        destination = './gta_python.zip'
        download_file_from_google_drive(file_id, destination)
        print('Unzipping Python trained model...')
        fz = zipfile.ZipFile("./gta_python.zip", 'r')
        for file in fz.namelist(): fz.extract(file, "./")
        # print('Clearing zip file...')     
        # os.remove("./data/gta_python.zip")
    print('Trained models downloaded.')