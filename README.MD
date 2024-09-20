
# Sudoku Solver using Python openCv and TensorFlow

This project collects computer vision techniques using OpenCV and TensorFlow, along with conventional Python algorithms, to detect and solve Sudoku puzzles.

This project was inpired by the work done by Aditi Jain [Aditi Jain repository for sudoku_AI](https://github.com/Joy2469/Sudoku_AI) Brenadan Artley [Medium Article Code for building mnist CNN](https://github.com/brendanartley/Medium-Article-Code/blob/main/README.md) and FreeCodeCamps sudoku solving with Python [Andvarjos Freecode camp sudoku solver](https://github.com/Andvarjo/freecodecamp/blob/main/Python/sudoku_solver.py).




## Development

For this project, we will use the OpenCV library to extract the Sudoku puzzle from the original image, preprocess it, and prepare it to extract each individual cell. Then, we will pass each extracted cell through a CNN (Convolutional Neural Network) to identify the number contained within and save this information. Finally, we will pass this data to a simple Python program that solves the Sudoku puzzle using a conventional algorithm.


### Process
![Process Drawing](sudoku/process.jpg)

### Steps


1. Image Preloading: The target image is preloaded, either from a locally stored file or captured from a camera. A script using OpenCV is attached for this purpose.

2. Image Preprocessing: The image is preprocessed to find the edges and corners of the largest area (the Sudoku puzzle). This allows us to crop and straighten the image from the original image.

3. Cell Extraction: Once the image is straightened, it is preprocessed to extract each individual cell of the Sudoku puzzle.

4. Cell Extraction: Each cell is extracted.

5. ROI Bounding Boxes: Within each cell, bounding boxes are found for the Region of Interest (ROI) that represents the number. This removes noise around the number and provides a clean, centered cell.

6. Cell Processing: Each cell is processed to match the dimensions, shape, and color characteristics of the MNIST dataset.

7. Convolutional Neural Network: Each cell is passed through a convolutional neural network to predict the corresponding number. Cells containing a blank space are assigned a value of zero to simplify the 
prediction process.

8. Digital Board Generation: The array of predictions is passed to the Sudoku script to display the digitized board on the screen.

9. Sudoku Solving: The Sudoku script solves the puzzle and displays the solved board.


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/Andvarjo/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/andres-varela92/)

