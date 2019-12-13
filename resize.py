"""
#Library https://pypi.org/project/google_images_download
#Code also adapted from https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
@author: Josh Pollock
"""
# import the necessary packages
from imutils.face_utils import FaceAligner
import os
import imutils
import dlib
import cv2
from google_images_download import google_images_download   #importing the library
#import selenium
from PIL import Image

def grab_images():
    """
    Uses google images download to grab images that fit the arguments
    """
#    browser = selenium.webdriver.Chrome("G:/PyCharm_2019.2.1/Anaconda3/Lib/site-packages/selenium/webdriver/chrome/chromedriver.exe")
    response = google_images_download.googleimagesdownload()   #class instantiation
#with selenium and a chromedriver it should be able to download more than 100 images at a time  #arguments = {"keywords":'Hyper-realistic face',"suffix_keywords":'drawing,painting,sculpture',"output_directory":'G:/Faces',"chromedriver":browser,"limit":5000,"print_urls":True,"format": "jpg"} 
    
    #If there are problems with the chromedriver this works.      
    arguments = {"keywords":'Hyper-realistic face',"suffix_keywords":'drawing,painting,sculpture',"output_directory":'G:/Faces',"print_urls":True,"format": "jpg"}   #creating list of arguments
    paths = response.download(arguments)   #passing the arguments to the function
    return(paths)   #passing absolute paths of the downloaded images
    img_dir = r"G:/Faces"
    for filename in os.listdir(img_dir):
        try :
            with Image.open(img_dir + "/" + filename):
                 print('ok')
        except :
            print(img_dir + "/" + filename)
            os.remove(img_dir + "/" + filename)

def crop_images(dirc):
    """
    Uses cv2 to crop images around the eyes
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("C:/Users/Josh Pollock/Google Drive/Senior/Senior Projects/Senior Project/shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    for dir, subdirs, files in os.walk(dirc):  #    Initially tried to iterate with os.chdir(dirc)
        for f in files:
            filename = f
            file = str(dirc+"/"+filename)
            image = cv2.imread(file)
            image = imutils.resize(image, width=256, height=256)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # show the original input image and detect faces in the grayscale
#            cv2.imshow("Input", image)
            rects = detector(gray, 2)
            for rect in rects:
                facealigned = fa.align(image, gray, rect)
                path = str('G:/Faces/Cropped/'+filename)
#                tr= str('C:/Users/Josh Pollock/Google Drive/Senior/Senior Projects/Senior Project/data/cropped/'+filename)
                cv2.imwrite(path, facealigned)
                cv2.waitKey(0)
    print("finished with batch")   
    return(filename)
    
def main():
#    grab_images()
    image = ["G:/Faces/Hyper-realistic face drawing","G:/Faces/Hyper-realistic face painting","G:/Faces/Hyper-realistic face sculpture"]
    for i in range(3): 
        print("starting batch") 
        dirc = image[i]
        crop_images(dirc)
    print("completely finished")            
main()