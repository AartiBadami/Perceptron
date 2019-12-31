import util

## Constants
DATUM_WIDTH = 0 # in pixels
DATUM_HEIGHT = 0 # in pixels

## Module Classes

class Datum:
  """
  A datum is a pixel-level encoding of digits or face/non-face edge maps.

  Digits are from the MNIST dataset and face images are from the 
  easy-faces and background categories of the Caltech 101 dataset.
  
  
  Each digit is 28x28 pixels, and each face/non-face image is 60x74 
  pixels, each pixel can take the following values:
    0: no edge (blank)
    1: gray pixel (+) [used for digits only]
    2: edge [for face] or black pixel [for digit] (#)
    
  Pixel data is stored in the 2-dimensional array pixels, which
  maps to pixels on a plane according to standard euclidean axes
  with the first dimension denoting the horizontal and the second
  the vertical coordinate:
    
    28 # # # #      #  #
    27 # # # #      #  #
     .
     .
     .
     3 # # + #      #  #
     2 # # # #      #  #
     1 # # # #      #  #
     0 # # # #      #  #
       0 1 2 3 ... 27 28
   
  For example, the + in the above diagram is stored in pixels[2][3], or
  more generally pixels[column][row].
       
  The contents of the representation can be accessed directly
  via the getPixel and getPixels methods.
  """
  def __init__(self, data,width,height):
    """
    Create a new datum from file input (standard MNIST encoding).
    """
    DATUM_HEIGHT = height
    DATUM_WIDTH=width
    self.height = DATUM_HEIGHT
    self.width = DATUM_WIDTH
    if data == None:
      data = [[' ' for i in range(DATUM_WIDTH)] for j in range(DATUM_HEIGHT)]
    self.pixels = util.arrayInvert(convertToInteger(data))

  def getPixel(self, column, row):
    """
    Returns the value of the pixel at column, row as 0, or 1.
    """
    return self.pixels[column][row]

  def getPixels(self):
    """
    Returns all pixels as a list of lists.
    """
    return self.pixels

  def getAsciiString(self):
    """
    Renders the data item as an ascii image.
    """
    rows = []
    data = util.arrayInvert(self.pixels)
    for row in data:
      ascii = map(asciiGrayscaleConversionFunction, row)
      rows.append( "".join(ascii) )
    return "\n".join(rows)

  def __str__(self):
    return self.getAsciiString()


# Data processing, cleanup and display functions

def loadDataFile(filename, n,width,height):
  """
  Reads n data images from a file and returns a list of Datum objects.
  
  (Return less then n items if the end of file is encountered).
  """
  DATUM_WIDTH=width
  DATUM_HEIGHT=height
  fin = readlines(filename)
  fin.reverse()
  items = []
  for i in range(n):
    data = []
    for j in range(height):
      data.append(list(fin.pop()))
    if len(data[0]) < DATUM_WIDTH-1:
      # we encountered end of file...
      print "Truncating at %d examples (maximum)" % i
      break
    items.append(Datum(data,DATUM_WIDTH,DATUM_HEIGHT))
  return items

import zipfile
import os
def readlines(filename):
  "Opens a file or reads it from the zip archive data.zip"
  if(os.path.exists(filename)):
    return [l[:-1] for l in open(filename).readlines()]
  else:
    z = zipfile.ZipFile('data.zip')
    return z.read(filename).split('\n')

def loadLabelsFile(filename, n):
  """
  Reads n labels from a file and returns a list of integers.
  """
  fin = readlines(filename)
  labels = []
  for line in fin[:min(n, len(fin))]:
    if line == '':
        break
    labels.append(int(line))
  return labels

def asciiGrayscaleConversionFunction(value):
  """
  Helper function for display purposes.
  """
  if(value == 0):
    return ' '
  elif(value == 1):
    return '+'
  elif(value == 2):
    return '#'

def IntegerConversionFunction(character):
  """
  Helper function for file reading.
  """
  if(character == ' '):
    return 0
  elif(character == '+'):
    return 1
  elif(character == '#'):
    return 2

def convertToInteger(data):
  """
  Helper function for file reading.
  """
  if type(data) != type([]):
    return IntegerConversionFunction(data)
  else:
    return map(convertToInteger, data)


import copy
import random

def perceptron_face_training(percent, iterations):

  n = (float(percent)/100) * 451
  n = int(n)
  items = loadDataFile("/Users/aarti/Desktop/ai_pa2/data/facedata/facedatatrain",451,60,70)
  labels = loadLabelsFile("/Users/aarti/Desktop/ai_pa2/data/facedata/facedatatrainlabels",451)

  # selecting n random images
  data_indexes = []
  for i in range(n):
    index = random.randint(0, 450)
    data_indexes.append(index)

  weight_vector = [0] * 100
  bias = 0

  for x in range(iterations):
    counter = 0 # current image
    # iterates over all training images
    for x in range(n):
      feature_vector = [0] * 100 # loads with count corresponding to curr image x
      i = 0
      j = 0
      for x in range(4200):
        if j == 60:
          j = 0
          i += 1
        if items[data_indexes[counter]].getPixel(j, i) != 0:
          index = (10 * (i//7)) + (j//6)
          feature_vector[index] += 1
        j += 1

      # calculating value of f(x) = dot product of features * weights + bias
      function = sum([x*y for x,y in zip(feature_vector,weight_vector)]) + bias
   
      if function >= 0 and labels[data_indexes[counter]] == 0:
        for x in range(100):
          weight_vector[x] -= feature_vector[x]
        bias -= 1
      elif function < 0 and labels[data_indexes[counter]] == 1:
        for x in range(100):
          weight_vector[x] += feature_vector[x]
        bias += 1
      else:
        random_useless_stuff = 0
      counter += 1

  return weight_vector, bias

def perceptron_face_testing(weight_vector, bias):
  n = 150
  items = loadDataFile("/Users/aarti/Desktop/ai_pa2/data/facedata/facedatatest",150,60,70)
  labels = loadLabelsFile("/Users/aarti/Desktop/ai_pa2/data/facedata/facedatatestlabels",150)
  counter = 0
  acc = 0

  for x in range(n): # iterates over all test images
    feature_vector = [0] * 100
    i = 0
    j = 0
    for x in range(4200): # loading features for current image
      if j == 60:
        j = 0
        i += 1
      if items[counter].getPixel(j,i) != 0:
        index = (10 * (i//7)) + (j//6)
        feature_vector[index] += 1
      j += 1

    function = sum([x*y for x,y in zip(feature_vector,weight_vector)]) + bias
    predicted_label = 1 if (function >= 0) else 0
    if predicted_label == labels[counter]: acc += 1
    counter += 1

  return (acc/150.0)
  
def perceptron_face_analysis():
  percentage = 10

  for x in range(10):
    i = 0
    print "acc at", percentage, "% training data : "
    while i < 10:
      weight_vector, bias = perceptron_face_training(percentage, 10)
      acc = perceptron_face_testing(weight_vector, bias)
      print acc
      i += 1
    percentage += 10

def perceptron_face_demo(image_index):
  weight_vector, bias = perceptron_face_training(100, 5)
  n = 150
  items = loadDataFile("/Users/aarti/Desktop/ai_pa2/data/facedata/facedatatest",150,60,70)
  labels = loadLabelsFile("/Users/aarti/Desktop/ai_pa2/data/facedata/facedatatestlabels",150)

  feature_vector = [0] * 100
  i = 0
  j = 0
  for x in range(4200): # loading features for image
    if j == 60:
      j = 0
      i += 1
    if items[image_index].getPixel(j,i) != 0:
      index = (10 * (i//7)) + (j//6)
      feature_vector[index] += 1
    j += 1

  function = sum([x*y for x,y in zip(feature_vector,weight_vector)]) + bias
  predicted_label = 1 if (function >= 0) else 0
  if predicted_label == labels[image_index]: print "correct prediction"
  else: print "incorrect prediction"
  print "predicted label : ", predicted_label, "| actual label : ", labels[image_index] 
  print items[image_index]

def perceptron_digit_training(percent, iterations):
  n = (float(percent)/100) * 5000
  n = int(n)
  items = loadDataFile("/Users/aarti/Desktop/ai_pa2/data/digitdata/trainingimages",5000,28,28)
  labels = loadLabelsFile("/Users/aarti/Desktop/ai_pa2/data/digitdata/traininglabels",5000)

  # selecting n random images
  data_indexes = []
  for i in range(n):
    index = random.randint(0, 4999)
    data_indexes.append(index)

  function = [0] * 10
  weight_vector = [[0]*49 for _ in range(10)]  
  bias = [0] * 10

  for x in range(iterations):
    counter = 0 # current image index
    for x in range(n): # iterates over all training images
      curr_label = labels[data_indexes[counter]]
      feature_vector = [[0]*49 for _ in range(10)] # loads with count
      i = 0
      j = 0
      for x in range(784):
        if j == 28:
          j = 0
          i += 1
        if items[data_indexes[counter]].getPixel(j, i) != 0:
          index = (7 * (i//4)) + (j//4)
          feature_vector[curr_label][index] += 1
        j += 1

      # calculating value of f(x) = dot product of features * weights + bias
      d = 0
      while d < 10:
        function[d] = sum([x*y for x,y in zip(feature_vector[d],weight_vector[d])]) + bias[d]
        d += 1

      predicted_label = function.index(max(function))
      correct_label = labels[data_indexes[counter]]
      counter += 1

      if predicted_label != correct_label:
        for x in range(49):
          weight_vector[predicted_label][x] -= feature_vector[predicted_label][x]
        bias[predicted_label] -= 1
        for x in range(49):
          weight_vector[correct_label][x] += feature_vector[correct_label][x]
        bias[correct_label] += 1 

  return weight_vector, bias


def perceptron_digit_testing(weight_vector, bias):
  n = 1000
  items = loadDataFile("/Users/aarti/Desktop/ai_pa2/data/digitdata/testimages",1000,28,28)
  labels = loadLabelsFile("/Users/aarti/Desktop/ai_pa2/data/digitdata/testlabels",1000)
  counter = 0
  acc = 0
  function = [0] * 10

  for x in range(n): # iterates over all test images
    i = 0
    j = 0
    # loading with feature count of current image
    feature_vector = [[0]*49 for _ in range(10)] # loads with count
    i = 0
    j = 0
    for x in range(784):
      if j == 28:
        j = 0
        i += 1
      if items[counter].getPixel(j, i) != 0:
        index = (7 * (i//4)) + (j//4)
        feature_vector[labels[counter]][index] += 1
      j += 1

    # calculating value of f(x) = dot product of features * weights + bias
    d = 0
    while d < 10:
      function[d] = sum([x*y for x,y in zip(feature_vector[d],weight_vector[d])]) + bias[d]
      d += 1

    predicted_label = function.index(max(function))
    correct_label = labels[counter]
    if predicted_label == correct_label: acc += 1
    counter += 1

  return (acc/1000.0)

def perceptron_digit_analysis():
  percentage = 10

  for x in range(10):
    i = 0
    print "acc at", percentage, "% training data : "
    while i < 10:
      weight_vector, bias = perceptron_digit_training(percentage, 1)
      acc = perceptron_digit_testing(weight_vector, bias)
      print acc
      i += 1
    percentage += 10

def perceptron_digit_demo(image_index):
  weight_vector, bias = perceptron_digit_training(100, 1)
  n = 1000
  items = loadDataFile("/Users/aarti/Desktop/ai_pa2/data/digitdata/testimages",1000,28,28)
  labels = loadLabelsFile("/Users/aarti/Desktop/ai_pa2/data/digitdata/testlabels",1000)
  function = [0] * 10
  feature_vector = [[0]*49 for _ in range(10)]

  i = 0
  j = 0
  for x in range(784): # loading features for image
    if j == 28:
      j = 0
      i += 1
    if items[image_index].getPixel(j,i) != 0:
      index = (7 * (i//4)) + (j//4)
      feature_vector[labels[image_index]][index] += 1
    j += 1

  # calculating value of f(x) = dot product of features * weights + bias
  d = 0
  while d < 10:
    function[d] = sum([x*y for x,y in zip(feature_vector[d],weight_vector[d])]) + bias[d]
    d += 1

  predicted_label = function.index(max(function))
  correct_label = labels[image_index]
  if predicted_label == correct_label: print "correct prediction"
  else: print "incorrect prediction"

  print "predicted label : ", predicted_label, "| actual label : ", correct_label
  print items[image_index]


# Testing
def _test():

  # perceptron_digit_analysis()
  # perceptron_face_analysis()

  ''' pick a number between 0-149 '''
  # perceptron_face_demo(0)

  ''' pick a number between 0-999 '''
  # perceptron_digit_demo(44)

if __name__ == "__main__":
  _test()


