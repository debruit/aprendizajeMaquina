
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


with open( sys.argv[ 1 ], 'r') as file:
    data = file.read( ).replace('\n', '')

util=data[ 53:]

print(util)
print(len(util))

aux=[[0 for i in range (0,100) ] for j in range (0,100) ]

for i in range (0,100):
    var=util[ 0: 100 ]
    for j in range (0,100):
        aux[i][j]=float(var[j])
    util=util[100:]

img=np.array(aux)

print(img)
c = plt.imshow(img, cmap ='Greens')
plt.show()


print(img[69][69])


ex=[[0 for i in range (0,3) ] for j in range (0,10000) ]

for i in range (0,100):
    for j in range (0,100):
        ex[i*100+j][2]=img[i][j]
        ex[i*100+j][0] =i
        ex[i*100+j][1] =j


#df = pd.DataFrame(ex)
#df.to_csv(r"C:\Users\Guatavita\Documents\semestre 14\ML\Codigos\PUJ_ML_01_en_clase\examples\python3\prueba.csv", index=False, header=False)

#print( util[ 0: 100 ] )
#print(type(data))
#print(data)
#print(util)
