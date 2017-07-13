
# coding: utf-8

# In[5]:

import urllib
import sys
import os

path = '../bugTestRed/'

VIRGA_KEY_1 = 'AIzaSyCwOOPCaQZOzeVQl-JKQi09sJn4jr8GqI8'
VIRGA_KEY_2 = 'AIzaSyA6DkXuCKVkZGqb1OqF—mGNE5PJxuLjMk'
# In[16]:

try:
    grads = ['0', '90', '180', '270']
    
    with open('../../bug_images_Rio_testred.pointList') as f:
        
        for k, line in enumerate(f.readlines()):
            #print line
            lat, lon = line.split(',')
            #print lat, lon
            lat = lat[1:]
            #lat = lat.strip(' ')
            lon = lon[1:-2]
            #lon = lon.strip(' ')
            
            print "Making request", lat, lon, "Of num:", k

            for c in grads:
                #print str(c)
                nome_arquivo = lat+'_'+lon+'_'+c+'.jpg'
                nome_arquivo_metadata = str(lat)+"_"+str(lon)+"_"+str(c)+".txt"
                if(not os.path.isfile(path+'images/'+nome_arquivo)):
                    urllib.urlretrieve("https://maps.googleapis.com/maps/api/streetview?size=640x640&location=" + str(lat) +","+ str(lon) + "&heading="+ str(c) +"&pitch=-0.76&key="+VIRGA_KEY_1, path+'images/'+nome_arquivo)
                else:
                    print "Already in here"
                if(not os.path.isfile(path+'meta/'+nome_arquivo_metadata)):
                    urllib.urlretrieve("https://maps.googleapis.com/maps/api/streetview/metadata?location="+ str(lat) +","+ str(lon) + "&key="+VIRGA_KEY_2,path+'meta/'+nome_arquivo_metadata)
                else:
                    print "Already in here"
except Exception,e: 
    print str(e)
    print("Usage: lat,lon cameraValue")


# In[ ]:



