#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 03:22:38 2019

@author: jetfire
"""

"""
Outcome ==> Scrapped 1123 Records Out of that 28 records are page load timeout
            or records was not avaliable. Used keyboard interrupt to stop at 
            1123 records.
Total Records => 1123 
Avaliable Records => 1095
Basic Analysis:
    Three Columns : Employee_Count, Linkedin_URL, Company_Name
    Employee_Count : 9 Categories
    Most Categories Count: 1001-5000
    Least Categories Count: 0-1
"""
from selenium import webdriver
from bs4 import BeautifulSoup
import getpass
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

hh = webdriver.ChromeOptions()
hh.add_argument("headless")
chrome_path = '/usr/bin/chromedriver'
driver = webdriver.Chrome(chrome_path)
driver.set_page_load_timeout(100000)

def Scrap():
    userid = str(input("Enter email address or number with country code: "))
    password = getpass.getpass('Enter your password:')
    driver.get("https://www.linkedin.com")
    driver.implicitly_wait(6)
    driver.find_element_by_xpath("""/html/body/nav/a[3]""").click()
    driver.find_element_by_xpath("""//*[@id="username"]""").send_keys(userid)
    driver.find_element_by_xpath("""//*[@id="password"]""").send_keys(password)
    driver.find_element_by_xpath("""//*[@id="app__container"]/main/div/form/div[3]/button""").click()
    driver.get("https://www.linkedin.com/directory/companies-a")
    connectionName = driver.find_element_by_xpath("""//*[@id="seo-dir"]/div/div[3]/div/ul""")
    data = BeautifulSoup(connectionName.get_attribute('innerHTML'), "lxml")
    A = [i['href'] for i in data.find_all('a', href=True) if i.text ]
    A = A[:1500] #taking 1st 1500 links
    B = []
    C = []
    NA = [] #Storing Values which did not load or dont have records.
    print(len(A))
    for i in A:
        try:
            driver.implicitly_wait(5)
            driver.get(i)
            driver.find_element_by_class_name("""org-page-navigation__items""")
            driver.find_element_by_link_text("About").click()
            driver.implicitly_wait(30)
            data1 = driver.find_element_by_class_name("""org-about-company-module__company-size-definition-text""")
            data2 = BeautifulSoup(data1.get_attribute('innerHTML'), "lxml")
            data2 = re.sub(r"[a-z\n]","",data2.string)
            C.append(data2)
            data0 = driver.find_element_by_class_name("org-top-card-summary__title")
            data1_2 = BeautifulSoup(data0.get_attribute('innerHTML'), "lxml")
            data1_2 = re.sub(r"[\n]","",data1_2.text)
            B.append(data1_2)
            print(len(C))
        except:
            print("Page load Timeout Occured. Quiting !!!")
            NA.append(i)
            print(len(NA))
            continue
    
    print(C)
    print(len(NA))
    A = [x for x in A if x not in NA] #removing Values from A which did not load.
    print(len(A))  
    return(A,B,C)
       

def save_to(x,y,z):
    df = pd.DataFrame({'Company_Name':y,'Linkedin_URL':x,'Employee_Count':z})
    print(df)
    df.to_csv('output.csv',index=False)    


def basic():
    df = pd.read_csv("output.csv")
    df.groupby(df['Employee_Count']).size()
    a4_dims = (10.2, 8)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.countplot(df['Employee_Count'])
    plt.show()

x,y,z = Scrap()       
save_to(x,y,z) 
basic() 







