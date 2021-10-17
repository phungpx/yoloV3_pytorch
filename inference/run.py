import cv2
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--pattern', type=str)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--output-dir', default='output/')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    image_paths = list(Path(args.input_dir).glob(args.pattern)) if args.pattern else [Path(args.input_dir)]

    config = utils.load_yaml(args.config)
    predictor = utils.create_instance(config)

    for i, image_path in enumerate(image_paths, 1):
        print('**' * 30)
        print(f'{i} / {len(image_paths)} - {image_path.name}')

        image = cv2.imread(str(image_path))
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

            if args.show:
                cv2.imshow(image_path.name, image)
                cv2.waitKey()
                cv2.destroyAllWindows()

        cv2.imwrite(str(output_dir.joinpath(image_path.name)), image)
