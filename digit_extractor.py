import numpy as np
import cv2



# this function allows us to border the cell with a white frame with an specific thickness to erase some noise of original borders
def frame(img,border,color=255):
  h,w=img.shape
  img[0:border,:]=color
  img[:,0:border]=color
  img[h-border:h,:]=color
  img[:,w-border:w]=color
  return img


#next we have to find a bounding box representing the ROI of the cell, containing the number or the blank space
#to erase all kind of noise in the image leaving us only with our component of interest
def digit_extractor(img):
  
  gray=img.copy()
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
  kernel = np.ones((3, 3), np.uint8)
  processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

  contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)#finding all contours

  #since every cell is 50x50, we wanto to know if something is inside the cell, if it isn't we want to draw
  #a Zero inside the cell
  sub_area=img[10:40,10:40]
  if np.all(sub_area == 255):

    #drawing the zero inside
    test = np.ones((28, 28), dtype="uint8") * 255  # white background
    center_coordinates = (14, 14)  # center of the image(28/2, 28/2)
    radius = 10  # radius simulating '0'
    color = (0)  # black color
    thickness = 2  # border thickness
    zero=cv2.circle(test, center_coordinates, radius, color, thickness)

    return zero
  #if something is inside the cell we want to know the bigest contour in the picture , that may represent the number iside
  else:
    if len(contours) > 1: #more than 1 contour found
      bigger = contours[1]
    else:
      bigger = contours[0] #only one contour meaning the whole cell was detected (no noise erased)
    x, y, w, h = cv2.boundingRect(bigger)
    image=img[y:y+h,x:x+w]
    return image



#once we have extracted our ROI we mus center the image an draw  the rest of the cell with the same size(50,50)
def centre_image(img,color=255):
  height, width = (50,50)
  top=(height-img.shape[0])//2
  bottom=height-img.shape[0]-top
  left=(width-img.shape[1])//2
  right=width-img.shape[1]-left

  centered= np.full((height, width),color)
  centered[top:top + img.shape[0],left:left + img.shape[1]] = img

  return centered







def cell_split(sudoku):
  height, width=sudoku.shape
  sudoku=cv2.threshold(sudoku, 128, 255, cv2.THRESH_BINARY)[1]
  
  sudoku_array=[]
  rows = 9
  cols = 9

  # Calculate the height and width of each cell
  cell_height = height // rows
  cell_width = width // cols

  # Loop through each cell and crop the image
  for i in range(rows):
      for j in range(cols):
          # Calculate the coordinates for cropping
          y_start = i * cell_height
          y_end = (i + 1) * cell_height
          x_start = j * cell_width
          x_end = (j + 1) * cell_width

          # Crop the image, erase additional info centre, resize and invert color since mnist have withe numbers and black background
          cell = sudoku[y_start:y_end, x_start:x_end]
          cell = frame(cell,5,255)
          cell = digit_extractor(cell)
          cell = centre_image(cell,255)
          cell = cell.astype('uint8')
          cell = cv2.resize(cell,(28,28))
          cell = cv2.bitwise_not(cell)

          # save the cropped cell
          sudoku_array.append(cell)
  return np.array(sudoku_array)