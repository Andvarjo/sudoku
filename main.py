import cv2
import numpy as np
import pre_process
import digit_extractor
import sudoku_solver
import tensorflow as tf
from tensorflow.keras.models import load_model



imagen_ruta = 'sudoku3.png' # for loading the sudoku image
img = pre_process.pre_grid(imagen_ruta) #obtain sudoku processed board image


sudoku_array = digit_extractor.cell_split(img) #obtain each cell


model =  tf.keras.models.load_model('mnist.h5')  #loading the model
sudoku_array = sudoku_array.astype('float32')/255 #shaping input data to
sudoku_array = sudoku_array.reshape(-1, 28, 28, 1) #shaping input data to

predictions=model.predict(sudoku_array)
puzzle=np.array(np.argmax(predictions,axis=1)) #predicting the cell values

puzzle = puzzle.reshape(9,9).tolist() #shaping array to a 9x9 list as an input

sudoku_solver.solve_sudoku(puzzle) #bring to console the original Board and the solution



