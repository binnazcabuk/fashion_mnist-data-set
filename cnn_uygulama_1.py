# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:15:44 2020

@author: ASUS
"""

# Gerekli kütüphaneler yüklenir
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

#Veri seti yuklenir
fashion_mnist = keras.datasets.fashion_mnist

""" Yukarıdaki satırda bir nesne hazırladık. 
İçindeki veriyi alabilmek için load_data metodunu kullanmamız gerekiyor. 
Eğer daha önceden veriyi indirmediysek bu metot bir indirme işlemi başlatacak."""

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#verinin boyutlarını inceleyelim.
print("Eğitim setinin boyutu: ")
print(train_images.shape)
print("Test setinin boyutu: ")
print(test_images.shape)

# Eğitim ve test setindeki eşsiz sınıf sayısı
unique_classes,u_counts = np.unique(np.concatenate([train_labels,test_labels]),return_counts=True)
print(unique_classes)
print(u_counts)


"""Toplamda 10 adet sınıf var, bu sınıfların hangi giysilere ait olduğu veri
setinin içinde yazmıyor. Bunun için sınıfların yazdığı bir liste oluşturalım."""

class_names = ['Tişört / Üst', 'Pantolon', 'Kazak', 'Elbise', 'Ceket',
               'Sandalet', 'Gömlek', 'Spor Ayakkabı', 'Çanta', 'Çizme']
#Her sınıftan kaç örnek göstereceğimiz bilgisi
num_of_samples_per_class = 10
#Kaç sınıfımız var
num_classes = len(unique_classes)

#verisetimizdeki ilk 10 resmi görelim
# plot first few images
for i in range(9):
	# define subplot
	plt.subplot(330 + 1 + i)
	# plot raw pixel data
	plt.imshow(train_images[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()
#Modelimizi olusturalim
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D
model=keras.Sequential()
model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=6, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(20, activation='softmax'))

#Modelimizdeki parametre sayısını öğrenelim
model.count_params()
#Modelimizi inceleyelim
model.summary()
#Modeli derleme
""" DİKKAT aşağıda model compile edilirken loss fonk. olarak 'sparse_categorical_crossentropy' kullanilmis. 
Bunun 'categorical_cross_entropy' den faki şöyle açıklanmış:

The only difference between sparse categorical cross entropy and categorical cross entropy is the format of true labels. 
When we have a single-label, multi-class classification problem, the labels are mutually exclusive for each data,
 meaning each data entry can only belong to one class. Then we can represent y_true using one-hot embeddings.
 
 
Seyrek kategorik çapraz entropi ve kategorik çapraz entropi arasındaki tek fark, gerçek etiketlerin formatıdır.
Tek etiketli, çok sınıflı bir sınıflandırma sorunumuz olduğunda, etiketler her veri için birbirini dışlar,
yani her veri girişi sadece bir sınıfa ait olabilir. Sonra tek sıcak düğünler kullanarak y_true temsil edebiliriz.
 
 yani ben etiketlerimi tek kolonda (0---9) ile gösterirsem 'sparse_categorical_crossentropy'
 her bir sınıf icin bir kolon oluşturup bir etiketi (ilk sınıfın etiketi söyle olur(1000000000) ) bu şekilde ifade edersem 
 'categorical_crossentropy' kullanirim.
 
 Link: https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
"""

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss='sparse_categorical_crossentropy',metrics=['accuracy'])

train_images = train_images.reshape(-1,28, 28, 1) 
test_images = test_images.reshape(-1,28, 28, 1) 
#Normalizasyonun önemini daha iyi anlatabilmek adına ilk önce imgeleri normalize etmeden modelimizi eğitelim
#model.fit(train_images, train_labels, epochs=5)
# Şimdi normalizasyon kullanarak yapalım
train_images = train_images / 255.0
test_images = test_images / 255.0
#Modeli eğitelim
model.fit(train_images, train_labels, epochs=12)

#Modelin değerlendirilmesi
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test seti \"accuracy\" değeri {:.2f}".format(test_acc))


