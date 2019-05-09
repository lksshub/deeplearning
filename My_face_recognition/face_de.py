
import tensorflow as tf 
import detect_face
import cv2
import numpy as np 


def img_de_test(img_name):
	img = cv2.imread(img_name)

	minsize = 20 # minimum size of face
	threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
	factor = 0.709 # scale factor

	sess = tf.Session()

	pnet_fun, rnet_fun, onet_fun = detect_face.create_mtcnn(sess,'./npy')

	results = detect_face.detect_face(img=img, minsize=minsize, pnet=pnet_fun, rnet=rnet_fun, onet=onet_fun, threshold=threshold, factor=factor)
	total_boxes = results[0] #人脸框信息
	points = results[1] #关键点信息

	print('total_boxes',total_boxes)

	if len(total_boxes) :

		print(points,total_boxes)
		draw = img.copy()

		cv2.rectangle(draw,(int(total_boxes[0][0]),int(total_boxes[0][1])),
						(int(total_boxes[0][2]),int(total_boxes[0][3])),(255,255,255))


		cv2.circle(img=draw,center=(int(points[0]),int(points[5])),radius=5,color=(0,0,255))
		cv2.circle(img=draw,center=(int(points[1]),int(points[6])),radius=5,color=(0,0,255))
		cv2.circle(img=draw,center=(int(points[2]),int(points[7])),radius=5,color=(0,0,255))
		cv2.circle(img=draw,center=(int(points[3]),int(points[8])),radius=5,color=(0,0,255))
		cv2.circle(img=draw,center=(int(points[4]),int(points[9])),radius=5,color=(0,0,255))


		# cv2.namedWindow('img',cv2.WINDOW_NORMAL)
		cv2.imshow('detection result',draw)

		cv2.waitKey(0)
		cv2.destroyAllWindows()

	else:
		print('没有人脸')
		return 0

			

def video_capture_de_test(video_path=0):

	minsize = 20 # minimum size of face
	threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
	factor = 0.709 # scale factor

	sess = tf.Session()
	pnet_fun, rnet_fun, onet_fun = detect_face.create_mtcnn(sess,'./npy')

	cap = cv2.VideoCapture(video_path)
	while(True):
		ret,frame = cap.read()
		# frame = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)

		results = detect_face.detect_face(img=frame, minsize=minsize, pnet=pnet_fun, rnet=rnet_fun, onet=onet_fun, threshold=threshold, factor=factor)

		total_boxes = results[0] #人脸框信息
		points = results[1] #关键点信息

		print(total_boxes,points)

		if len(total_boxes) :
			cv2.rectangle(frame,(int(total_boxes[0][0]),int(total_boxes[0][1])),
							(int(total_boxes[0][2]),int(total_boxes[0][3])),(255,255,255))
			cv2.circle(img=frame,center=(int(points[0]),int(points[5])),radius=5,color=(0,0,255))
			cv2.circle(img=frame,center=(int(points[1]),int(points[6])),radius=5,color=(0,0,255))
			cv2.circle(img=frame,center=(int(points[2]),int(points[7])),radius=5,color=(0,0,255))
			cv2.circle(img=frame,center=(int(points[3]),int(points[8])),radius=5,color=(0,0,255))
			cv2.circle(img=frame,center=(int(points[4]),int(points[9])),radius=5,color=(0,0,255))
			cv2.imshow('detection result',frame)

		else:
			cv2.imshow('detection result',frame)
			print('没有人脸')

		if cv2.waitKey(1) &0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	video_capture_de_test()