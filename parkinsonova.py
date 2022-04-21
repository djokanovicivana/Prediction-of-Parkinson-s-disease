#UCITAVANJE BIBLIOTEKA
import numpy as np #za nizove
import pandas as pd #ako su podaci u tabelama, da mozemo lako da im pristupamo
from sklearn.model_selection import train_test_split #podatke koje imamo treba da rastavimo na train data i test data
from sklearn.preprocessing import StandardScaler #za procesovanje podataka koje koristimo
from sklearn import svm #za support vector machine koji koristimo ovde 
from sklearn.metrics import accuracy_score #da odredimo koliko nam je dobar model 
#UPRAVLJANJE PODACIMA
#ucitavanje podataka iz csv fajla u Panda DataFrame
podaci=pd.read_csv('D:\parkinsons\parkinsons.csv')
print('Broj osoba i parametara u data setu: ' ,podaci.shape) #daje broj redova i kolona u dataframe-u
print('Broj osoba u data setu koje imaju i koje nemaju Parkinsonovu bolest: ',podaci['status'].value_counts()) #status nam je target promenljiva(1 je ako osoba ima Parkinsona, a 0 ako osoba nema). Ovom funkcijom odredjujemo koliko osoba u nasem datasetu ima Parkinsona, a koliko osoba nema
print('Prosecne vrednosti parametara grupisane na osnovu toga da li osoba ima Parkinsonovu bolest ili ne:\n', podaci.groupby('status').mean()) #grupisemo podatke po status promenljivoj, i radimo srednju vrednost za sve osobe koje imaju parkinsona, i za sve osobe koje nemaju, na osnovu toga posle treniramo model da odredimo za neke vrednosti da li su od osobe koje imaju parkinsona ili ne 
#DATA PRE-PROCESSING 
A = podaci.drop(columns=['name','status'], axis=1) #razdvajanje kolone name i status od ostalih kolona iz data seta, 1 znaci da odvajamo kolonu, a 0 bi bila za red, name nam ne treba
B = podaci['status'] #smestamo kolonu status u posebnu promenljivu
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.25, random_state=3) #delimo podatke u training set i test set, koristimo funkciju train_test_split, test_size pokazuje da ce test deo da nam ima 20%, a training deo 80%, random_state pokazuje 
#STANDARDIZOVANJE PODATAKA
s = StandardScaler()
s.fit(A_train) #fitujemo samo training skup, ne i test
A_train = s.transform(A_train) # posto su parametri naseg data seta u razlicitim opsezima, potrebno je da ih sve transformisemo tako da budu isti opseg

A_test = s.transform(A_test) # to radimo i sa training i sa test skupom
#TRENIRANJE MODELA

#Support Vector Machine Model- potrebno je pronaci hyperplane, to je kao linija koja treba da razdvoji sve podatke u neke dve grupe
m = svm.SVC(kernel='linear')
#treniranje modela na trening skupu 
m.fit(A_train, B_train) # na ovaj nacin smo istrenirali nas  model, treniranje ide brzo jer imamo malo podataka, tj. mali nam je data set 
#Model Evaluation

#Accuracy Score #odredjuje koliko nam je dobar model, tj koliko procenata moze dobro da predvidi da li osoba ima parkinsona ili ne
# accuracy score on training data
A_train_predvidjanje = m.predict(A_train)
verodostojnost_training = accuracy_score(B_train, A_train_predvidjanje)
# radimo accuracy score i na training i na test setu
print('Verodostojnost trening skupa : ', verodostojnost_training)
# accuracy score on training data
A_test_predvidjanje = m.predict(A_test)
verodostojnost_test = accuracy_score(B_test, A_test_predvidjanje)
print('Verodostojnost test skupa : ',verodostojnost_test )
#poenta je da accuracy score, i za jedan i za drugi set bude preko 75% da bi se reklo da nam model dobro radi 
#takodje je poenta da razlika izmedju accuracy scora training i test seta bude minimalna, u suprotnom dolazi do overfittinga ili underfittinga?

#Building a Predictive System
provera = (162.56800,198.34600,77.63000,0.00502,0.00003,0.00280,0.00253,0.00841,0.01791,0.16800,0.00793,0.01057,0.01799,0.02380,0.01170,25.67800,0.427785,0.723797,-6.635729,0.209866,1.957961,0.135242) #testiramo na jednom specificnom primeru da li nam model radi, iz baze kopiramo sve kolone osim status kolone

# podatke na  kojima proveravamo pretvaramo u numpy array
provera_np = np.asarray(provera) 

# reshape the numpy array
reshaped = provera_np.reshape(1,-1) # ukoliko ovo ne uradimo, model ce misliti da smo mu prosledili sve podatke iz baze, a ne samo jedan 

# standardizovanje podataka
standardizovano = s.transform(reshaped) #stavljamo sve podatke da budu u istom opsegu, tj sve parametre iz baze

predvidjanje = m.predict(standardizovano) # pravimo promenljivu koja ce da nam predstavlja predvidjanje i koristimo model promenljivu, tj koristimo nas model koji je istreniran
print(predvidjanje) #predvidjanje zapravo vraca samo vrednost 1 ili 0.


if (predvidjanje[0] == 1): #predvidjanje je zapravo lista, a mi imamo samo jednog pacijenta, tako da mu pristupamo preko indeksa 0
  print("Ova osoba ima Parkinsonovu bolest.")

else:
  print("Ova osoba nema Parkinsonovu bolest.")
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

