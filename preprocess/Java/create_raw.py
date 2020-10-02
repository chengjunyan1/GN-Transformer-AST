import json,os,requests,zipfile


tlcs_url = "https://github.com/xing-hu/TL-CodeSum/archive/master.zip"
r = requests.get(tlcs_url) 
with open("./TL-CodeSum-master.zip",'wb') as f: f.write(r.content)
fz = zipfile.ZipFile("./TL-CodeSum-master.zip", 'r')
for file in fz.namelist(): fz.extract(file, "./")       
os.remove("./TL-CodeSum-master.zip")

path='./TL-CodeSum-master/data/'
save='./Raw/'
if not os.path.exists(save): os.makedirs(save)

for mode in ['test','train','valid']:
    coms={}
    with open(path+mode+'/'+mode+'.token.nl','r',encoding='utf8') as f:
        q=[i.rstrip().split('\t') for i in f.readlines()]
    for i in q: coms[i[0]]=i[1].split(' ')
    srcs={}
    with open(path+mode+'/'+mode+'.json','r',encoding='utf8') as f:
        w=[json.loads(i.rstrip()) for i in f.readlines()]
    for i in w:
        srcs[str(i['id'])]=i['code']
    with open(save+mode+'_com.json','w') as f: json.dump(coms,f)
    with open(save+mode+'_src.json','w') as f: json.dump(srcs,f)