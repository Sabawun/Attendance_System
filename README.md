# Attendance_System
  Student Attendance System using face recognition. 

Training_Testing
   Images are converted into np.array, normalized and flatten. Final shape, with labels is (1, 12288)
   Each image is 64x64 so that's 4096 pixels in total, considering the RGB factor (3) -> 4096 * 3 makes 12288 hence the label is (1,12288)

PCA_Attendance 
   Training & Testing Images converted into PCA
  
CNN (Rename NN)
   PCA form training images forwarded into neural network model (sequential), model saved as "Face_Recognizer". 

Mobile 
   Mobile User Interface

This is the change enver hocam's looking for.
