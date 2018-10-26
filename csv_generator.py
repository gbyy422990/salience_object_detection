#coding:utf-8
import os
import csv

def create_csv(dirname):
    path = './dataset/'+ dirname +'/'
    name = os.listdir(path)
    #name.sort(key=lambda x: int(x.split('.')[0]))
    #print(name)
    with open (dirname+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['data', 'label'])
        for n in name:
            if n[-4:] == '.jpg':
                print(n)
                # with open('data_'+dirname+'.csv','rb') as f:
                writer.writerow(['./dataset/'+str(dirname) +'/'+ str(n),'./dataset/' + str(dirname) + 'label/' + str(n[:-4] + '.png')])
            else:
                pass

if __name__ == "__main__":
    create_csv('misc')
    create_csv('misctest')
