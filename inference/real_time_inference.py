import cv2
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import utils


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='config.yaml')
	args = parser.parse_args()

	cap = cv2.VideoCapture(0)

	cap.set(3, 640)
	cap.set(4, 480)

	config = utils.load_yaml(args.config)
	predictor = utils.create_instance(config)

	

	while True:
		success, image = cap.read()
		prediction = predictor([image])[0]

		if prediction['labels'] is not None:
		    thickness = max(image.shape) // 500
		    fontscale = max(image.shape) / 500
		    boxes = prediction['boxes'].cpu().numpy().astype(np.int32)
		    labels = prediction['labels'].cpu().numpy()
		    scores = prediction['scores'].cpu().numpy()
		    class_names = prediction['names']
		    for box, score, class_name in zip(boxes, scores, class_names):
		        color = (
		            np.random.randint(200, 255),
		            np.random.randint(50, 200),
		            np.random.randint(0, 150)
		        )

		        cv2.rectangle(
		            img=image,
		            pt1=tuple(box[:2]),
		            pt2=tuple(box[2:]),
		            color=color,
		            thickness=thickness
		        )

		        cv2.putText(
		            img=image,
		            text=f'{class_name}: {score: .4f}',
		            org=tuple(box[:2]),
		            fontFace=cv2.FONT_HERSHEY_PLAIN,
		            fontScale=fontscale,
		            color=color,
		            thickness=thickness,
		            lineType=cv2.LINE_AA)

		cv2.imshow("Output", image)	

		if cv2.waitKey(1) == ord('q'):
			break	
	cv2.waitKey(1)
	cv2.destroyAllWindows()
