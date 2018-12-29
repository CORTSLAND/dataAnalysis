# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:05:35 2018

@author: cortm
"""



from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

from scipy.stats import mode

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import numpy as np

import seaborn as sns






''' PART 1'''
#Load the digits dataset
digits = load_digits()
data = digits.data
label = digits.target #truth lables
images = digits.images

#kmeans analysis
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)





'''Part 2'''

#zip images and labels together
labledimgs = list(zip(digits.images, digits.target))

#random generator of 4x4 numbers
plt.figure(figsize=(6,7))
for index, (image, label) in enumerate(labledimgs[:16]):
    plt.subplot(4,4,index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % label)







'''PART 3'''

#set data with its actual label
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]
   
#accuracy of the data    
acc = accuracy_score(digits.target, labels)

#create confusion matrix and apply heatmap with labels
confmat = confusion_matrix(label, labels)
sns.heatmap(confmat.T, square=True, annot=True, fmt='d', cbar=False,xticklabels=digits.target_names,yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title('Confusion Matrix; Accuracy = %.2f' %acc + '%',color='red')





'''Part 4'''

#Elbow finding K
iner = {}
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k).fit(data)
    newest = kmeans.labels_
    iner[k] = kmeans.inertia_
 
#use log10 for y-axis and plot markers
log = np.log10(list(iner.values()))  
X = range(1,15)
plt.figure(figsize=(9,4))
plt.plot(X, log, '-bo',markevery=1)

#create background
ax = plt.gca()
ax.set_facecolor('aliceblue')

#labels for axis and certain areas on plot
plt.xlabel("number of clusters, k")
plt.ylabel("log10 Inertia")
plt.xticks(range(1,15))
plt.text(9.5,6.09,'K=10',color='red')
plt.grid(True,color='white',linewidth=3)
plt.show()







'''Part 5'''


#label each number to its label vertivally and horizontially
x = np.vstack([digits.data[digits.target==i]
               for i in range(10)])
y = np.hstack([digits.target[digits.target==i]
               for i in range(10)])
   
 #T-SNE algorithm on the data    
snee = TSNE(n_components=2).fit_transform(x)

#Change color for each num
difcolor = np.array(sns.color_palette("hls", 10))

#plot the figure
plt.figure(figsize=(4.5, 4.5))
ax = plt.subplot(aspect='equal')
plt.scatter(snee[:,0], snee[:,1], s=10,c=difcolor[y.astype(np.int)])
plt.xlim(-80, 80)
plt.ylim(-60, 80)

#attach the correct numbers to its cluster
nums = []
for q in range(10):
    xnumb, ynumb = np.median(snee[y == q, :], axis=0)
    num = ax.text(xnumb, ynumb, str(q), fontsize=24)
    num.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"),PathEffects.Normal()])
    nums.append(num)

    