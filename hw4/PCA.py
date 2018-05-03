from skimage import io
import numpy as np
import sys
import os

def readFile(imagePath):
    images = []
    for i in range(415):
    # for f in os.listdir(imagePath):
        filename = (imagePath +'/' + str(i) + '.jpg')
        # filename = os.path.join(imagePath, f)
        print(filename)
        picture = io.imread(filename)
        picture = np.reshape(picture, 1080000)
        images.append(picture)
    return images

def saveImage(fileName, data):
    data -= np.min(data)
    data /= np.max(data)
    data = data.reshape((600, 600, 3))
    io.imsave(fileName, (data*255).astype(np.uint8))

def averageFace(images):
    print('======[generating averageface]=====')
    images = np.array(images)
    print(images.shape)
    meanFace = np.mean(images, axis=0)
    meanFace = np.reshape(meanFace, (600,600,3)).astype(np.uint8)
    io.imsave('avg.jpg', meanFace)

def eigenFace(rawdata):
    print('======[generating eigenface]=====')
    mean = np.mean(rawdata, axis=0)
    data = rawdata - mean

    u, w, v = np.linalg.svd(data, full_matrices=False)
    summation = np.sum(w)
    w = w * 100 / summation
    print("eigenvalue" , w.shape)
    print("eigenvector", v.shape)
    for i in range(4):
        print(round(w[i], 1))
        # eigenFace = np.reshape( v[i], (600, 600, 3))   
        # eigenFace -= np.min(eigenFace)
        # eigenFace /= np.max(eigenFace)
        # eigenFace = (eigenFace * 255).astype(np.uint8)

        fileName = 'eigen' + str(i+1) + '.jpg'
        saveImage(fileName, v[i])
        # io.imsave( fileName, eigenFace)

def reconstruct(images, index):
    print('======[reconstructing face]=====')
    mean = np.mean(images, axis=0)
    images = images - mean
    u, w, v = np.linalg.svd(images, full_matrices=False)
    S = np.diag(w)
   
    u_ = u[:, :4].dot(S[:4, :4])
    eigen = v[0:4]
    Recon = u_.dot(eigen) + mean
    print(Recon.shape)
    saveImage('reconstruction.jpg', Recon[int(index)-1])

    
if __name__ == '__main__':   
    images = readFile(sys.argv[1])
    eigenFace(images)
    selected = sys.argv[2].split('.')[0]
    reconstruct(images, selected)


