import os,requests,zipfile,gzip,shutil


# tlcs_url = "https://github.com/EdinburghNLP/code-docstring-corpus/archive/master.zip"
# r = requests.get(tlcs_url) 
# with open("./code-docstring-corpus-master.zip",'wb') as f: f.write(r.content)
# fz = zipfile.ZipFile("./code-docstring-corpus-master.zip", 'r')
# for file in fz.namelist(): fz.extract(file, "./")       
# os.remove("./code-docstring-corpus-master.zip")

path='./code-docstring-corpus-master/parallel-corpus'
save='./Raw/'
for mode in ['test','train','valid']:
    if not os.path.exists(save+mode): os.makedirs(save+mode)
    if mode=='train':
        with gzip.open(path+'/data_ps.declbodies.'+mode+'.gz', 'rb') as f_in:
            with open(save+'/'+mode+'/data_ps.declbodies.'+mode, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        source_file=path+'/data_ps.declbodies.'+mode
        destination_file=save+'/'+mode+'/data_ps.declbodies.'+mode
        shutil.copyfile(source_file, destination_file)
    source_file=path+'/data_ps.descriptions.'+mode
    destination_file=save+'/'+mode+'/data_ps.descriptions.'+mode
    shutil.copyfile(source_file, destination_file)