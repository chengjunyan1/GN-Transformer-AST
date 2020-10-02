import pickle,json,time,os


def readlines(path):
    with open(path,'r') as f:
        q=f.readlines()
    return q

def loaddata(path):
    with open(path,'r') as f:
        q=f.read()
    return eval(q)

def savejson(d,path):
    json_str = json.dumps(d)
    with open(path,'w') as f: 
        f.write(json_str)
    
def savejsonb(d,path):
    json_str = json.dumps(d)
    with open(path,'wb') as f: 
        f.write(json_str)
        
def loadjson(path):
	with open(path, 'r') as f:
		d = json.load(f)
	return d

def loadjsonb(path):
	with open(path, 'rb') as f:
		d = json.load(f)
	return d

def savepkl(d,path):
    with open(path,'wb') as f: 
        pickle.dump(d,f)
        
def loadpkl(path): 
    with open(path,'rb') as f: 
        d = pickle.load(f)
    return d

def loadpid(path):
	data = {}
	for line in open(path, 'r').readlines()[1:]:
		t = line.split('\t')
		data[int(t[0])]=int(t[1])
	return data


    
    
    