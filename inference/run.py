import cv2
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--image-paths', nargs='+')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--output-dir', default='output/')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    config = utils.load_yaml(args.config)
    predictor = utils.create_instance(config)

    images = [cv2.imread(image_path) for image_path in args.image_paths]
    predictions = predictor(images)

    for i in range(len(images)):
        if predictions[i]['labels'] is not None:
            thickness = max(images[i].shape) // 500
            fontscale = max(images[i].shape) / 500
            boxes = predictions[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = predictions[i]['labels'].cpu().numpy()
            scores = predictions[i]['scores'].cpu().numpy()
            class_names = predictions[i]['names']
            for box, score, class_name in zip(boxes, scores, class_names):
                color = (
                    np.random.randint(200, 255),
                    np.random.randint(50, 200),
                    np.random.randint(0, 150)
                )

                cv2.rectangle(
                    img=images[i],
                    pt1=tuple(box[:2]),
                    pt2=tuple(box[2:]),
                    color=color,
                    thickness=thickness
                )

                cv2.putText(
                    img=images[i],
                    text=f'{class_name}: {score: .4f}',
                    org=tuple(box[:2]),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=fontscale,
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA)

                if args.show:
                    print('OK')
                    plt.imshow(images[i])
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()

        # cv2.imwrite(str(output_dir.joinpath(Path(args.image_paths[i]).name)), images[i])
        # plt.imshow(image)
