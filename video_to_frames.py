import os
folder = 'test'  
os.mkdir(folder)
import cv2
#print(cv2.__version__)
vidcap = cv2.VideoCapture('test_car.avi')
count = 0
while True:
    success,image = vidcap.read()
    if not success:
        break
   # cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), image)     # save frame as JPEG file
    cv2.imwrite(os.path.join(folder,"{:04d}.jpg".format(count)), image)     
    count += 1
print("{} images are extacted in {}.".format(count,folder))
