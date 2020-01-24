# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 12:27:00 2020

@author: Madhumita Shankar
"""

from newsapi import NewsApiClient
import pandas as pd

# Init
newsapi = NewsApiClient(api_key='ef53b50375694b959459123644c0a627')

sources = newsapi.get_sources()
sources.keys()

o,n,d,u,c,l,co = [],[],[],[],[],[],[]
for i in sources['sources']:
    o.append(i['id'])
    n.append(i['name'])
    d.append(i['description'])
    u.append(i['url'])
    c.append(i['category'])
    l.append(i['language'])
    co.append(i['country'])
    
df = pd.DataFrame({"id":o,"name":n,"des":d,"url":u,"cat":c,"lan":l,"country":co})    
names = df['name'].to_list()
ids = df['id'].to_list() 
gnews = []
for k,v in zip(names,ids,):
    if "Google" in k:
        gnews.append(v)
gn = ",".join(gnews)
query = "Xanadu"   
all_articles = newsapi.get_everything(q=query,
                                      sources=['bbc-news'],
                                      domains='bbc.co.uk,techcrunch.com,news.google.com',
                                      to='2019-12-31',
                                      language='en',
                                      sort_by='relevancy',
                                      page=1)
    
    
    
    