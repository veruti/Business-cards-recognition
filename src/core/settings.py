from pydantic import BaseConfig


class Settings(BaseConfig):
    OPENVINO_TEXT_DETECTION_3_MODEL = (
        "models/openvino/text_detection/text-detection-0003.xml"
    )
    OPENVINO_TEXT_DETECTION_3_CONFIG = (
        "models/openvino/text_detection/text-detection-0003.bin"
    )


settings = Settings()
