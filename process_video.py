import cv2
import sys
import argparse
import os
import re
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    filename = re.search('\d{1,2}', pathIn).group(0)
    vidcap = cv2.VideoCapture(pathIn)
    """
    while not vidcap.isOpened():
        vidcap = cv2.VideoCapture('/home/testJD/data/Pig_Identification_Qualification_Train/train/1.mp4')
        cv2.waitKey(1000)
    """
    success,image = vidcap.read()


    totalFrameNumber = vidcap.get(cv2.CAP_PROP_FPS)
    print("This video has {} frames".format(totalFrameNumber))

    count = 0
    success = True
    currentPath = os.getcwd()
    imagesFolder = os.path.join(currentPath,filename)
    if not os.path.exists(imagesFolder):
        os.mkdir(imagesFolder)

    while success:
      success,image = vidcap.read()
      print("Read a new frame:", success)
#      image = cv2.resize(image,(512,384))
      cv2.imwrite("%s/%d.jpg" % (imagesFolder,count), image)     # save frame as JPEG file
      count += 1

if __name__ == "__main__":
    print("video processing begins:")
    arg = argparse.ArgumentParser()
    arg.add_argument("--pathIn", help="path of video")
    arg.add_argument("--pathOut", help="path to storre images")
    args = arg.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut)


















#
# import cv2
# import sys
# import argparse
#
# print(cv2.__version__)
#
# def extractImages(pathIn, pathOut):
#     vidcap = cv2.VideoCapture(pathIn)
#     success,image = vidcap.read()
#     totalFrameNumber = vidcap.get(cv2.CAP_PROP_FPS)
#     print("This video has {} frames".format(totalFrameNumber))
#     count = 0
#     success = True
#     while count < 5:
#       success,image = vidcap.read()
#       print("Read a new frame:", success)
#       image = cv2.resize(image,(512,384))
#       cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
#       count += 1
#
# if __name__ == "__main__":
#     print("video processing begins:")
#     arg = argparse.ArgumentParser()
#     arg.add_argument("--pathIn", help="path ot video")
#     arg.add_argument("--pathOut", help="path to storre images")
#     args = arg.parse_args()
#     print(args)
#     extractImages(args.pathIn, args.pathOut)
#
#
# import cv2
#
# cap = cv2.VideoCapture("1.mp4")
# while not cap.isOpened():
#     cap = cv2.VideoCapture("1.mp4")
#     cv2.waitKey(1000)
#     print("Wait for the header")
#
# pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
# count = 0
# while True:
#     flag, image = cap.read()
#     if flag:
#         # The frame is ready and already captured
#         # cv2.imshow('video', frame)
#         pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
#         print("now read" + str(pos_frame)+" frames")
#         # print("Read a new frame:", success)
#         image = cv2.resize(image, (512, 384))
#         cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
#         count += 1
#     else:
#         # The next frame is not ready, so we try to read it again
#         cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
#         print ("frame is not ready")
#         # It is better to wait for a while for the next frame to be ready
#         cv2.waitKey(1000)
#
#     if cv2.waitKey(10) == 27:
#         break
#     if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
#         # If the number of captured frames is equal to the total number of frames,
#         # we stopimport cv2
# import sys
# import argparse
#
# print(cv2.__version__)
#
# def extractImages(pathIn, pathOut):
#     vidcap = cv2.VideoCapture(pathIn)
#     success,image = vidcap.read()
#     totalFrameNumber = vidcap.get(cv2.CAP_PROP_FPS)
#     print("This video has {} frames".format(totalFrameNumber))
#     count = 0
#     success = True
#     while count < 5:
#       success,image = vidcap.read()
#       print("Read a new frame:", success)
#       image = cv2.resize(image,(512,384))
#       cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
#       count += 1
#
# if __name__ == "__main__":
#     print("video processing begins:")
#     arg = argparse.ArgumentParser()
#     arg.add_argument("--pathIn", help="path ot video")
#     arg.add_argument("--pathOut", help="path to storre images")
#     args = arg.parse_args()
#     print(args)
#     extractImages(args.pathIn, args.pathOut)
