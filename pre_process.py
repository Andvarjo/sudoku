import numpy as np
import cv2


#first we want to load the image using opencv and convert it to grayscale
def read_image(img_route):
  img = cv2.imread(img_route)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img

#we must prepare the image in order to detect borders and to warp it,
#so we need to aply some filters
def preprocess_img(img):

  preproc = cv2.GaussianBlur(img.copy(), (9, 9), 0) #borders better definition
  preproc = cv2.adaptiveThreshold(img.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) #apply treshold
  preproc = cv2.bitwise_not(preproc, preproc) #invert color
  kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)# dilate borders
  
  preproc = cv2.dilate(preproc, kernel,iterations=1)
  return preproc

#we need to find the corners of the sudoku board to be able to crop it and warp it if the picture is not straight
def find_corners(preproc):
  ext_contours,_ = cv2.findContours(preproc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find all contours

  ext_contours = sorted(ext_contours, key=cv2.contourArea, reverse=True)
  for c in ext_contours:

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    if len(approx) == 4: #finding biggest contours with 4 elements
      final=approx
      break

  corners = [(corner[0][0], corner[0][1]) for corner in final]
  return corners


#sometimes the coordenates that cv2 provide us are not in the right order so we have tu order them
def find_position(corners):   #function to order coordenates
  corners=sorted(corners ,key=lambda x: x[0] )
  left=corners[0],corners[1]
  right=corners[2],corners[3]
  if left[0][1]<left[1][1]:
    top_l=left[0]
    bottom_l=left[1]
  else:
    top_l=left[1]
    bottom_l=left[0]
  if right[0][1]<right[1][1]:
    top_r=right[0]
    bottom_r=right[1]
  else:
    top_r=right[1]
    bottom_r=right[0]
  return top_l,top_r,bottom_l,bottom_r

#now we can trasform the image well centered and warped, cleaning all the data we dont need from the original pic
#provideng us only the sudoku board
def transform(img,corners):

  top_l, top_r, bottom_l, bottom_r = find_position(corners)
  ordered_corners = top_l,top_r,bottom_r,bottom_l

      # Determine width of new img which is the max distance between
      # (bottom right and bottom left) or (top right and top left) x-coordinates
  width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
  width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
  width = max(int(width_A), int(width_B))

      # Determine height of new img which is the max distance between
      # (top right and bottom right) or (top left and bottom left) y-coordinates
  height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
  height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
  height = max(int(height_A), int(height_B))

      # Construct new points to obtain top-down view of img in
      # top_r, top_l, bottom_l, bottom_r order
  dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                            [0, height - 1]], dtype="float32")

      # Convert to Numpy format
  ordered_corners = np.array(ordered_corners, dtype="float32")

      # calculate the perspective transform matrix and warp
      # the perspective to grab the screen
  grid = cv2.getPerspectiveTransform(ordered_corners, dimensions)

      # Return the transformed img
  transformed=cv2.warpPerspective(img, grid, (width, height))
  
  return transformed

#our final step is to apply all previous transformations to the original image
def pre_grid(imagen_ruta):
  img=read_image(imagen_ruta)
  preproc=preprocess_img(img)
  corners=find_corners(preproc)
  sudoku=transform(img,corners)
  sudoku= cv2.resize(sudoku,(450,450))

  return sudoku


