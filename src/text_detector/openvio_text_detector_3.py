import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from openvino.inference_engine import IECore
from src.core.settings import settings
from src.text_detector.base import BaseDetector


class OpenVinoTextDetector3(BaseDetector):
    def __init__(self, conf_threshold=0.9):
        self.ie = IECore()
        self.network = self.ie.read_network(
            model=settings.OPENVINO_TEXT_DETECTION_3_MODEL,
            weights=settings.OPENVINO_TEXT_DETECTION_3_CONFIG,
        )
        self.executable_network = self.ie.load_network(
            self.network, device_name="CPU", num_requests=1
        )
        self.input_name = next(iter(self.network.input_info))

        self.conf_threshold = conf_threshold

        self.input_height, self.input_width = (768, 1280)

    def preprocess_image(self, image: np.array):
        resized = cv.resize(
            src=image,
            dsize=(self.input_width, self.input_height),
            interpolation=cv.INTER_LINEAR,
        )
        return resized.transpose(2, 0, 1)

    def predict(self, image: np.array):

        original_height, original_width = image.shape[:2]
        nn_input = self.preprocess_image(image=image)

        nn_outputs = self.executable_network.infer({self.input_name: nn_input})

        segm_logits = np.moveaxis(nn_outputs["model/segm_logits/add"], 1, 3)
        link_logits = np.moveaxis(nn_outputs["model/link_logits_/add"], 1, 3)

        sf_segm = np.exp(segm_logits[0][:, :, 1]) / np.sum(np.exp(segm_logits), axis=-1)

        # for i in range(16):
        #     sf_segm = np.exp(link_logits[0][:, :, i]) / np.sum(
        #         np.exp(link_logits), axis=-1
        #     )

        plt.imshow(sf_segm[0] > 0.9)
        plt.savefig(f"123.png")

        # faces_raw = nn_outputs["detection_out"][0][0]

        # faces = []
        # for (_, label, conf, x_min, y_min, x_max, y_max) in faces_raw:
        #     if self.conf_threshold <= conf:
        #         x_min = abs(int(x_min * original_width))
        #         x_max = abs(int(x_max * original_width))
        #         y_min = abs(int(y_min * original_height))
        #         y_max = abs(int(y_max * original_height))

        #         faces.append([x_min, y_min, x_max, y_max])

        # return faces
