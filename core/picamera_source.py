import time

from picamera2 import Picamera2


class Picamera2Source:
    """Direct IMX708 camera access through libcamera/Picamera2."""

    def __init__(self, width=1920, height=1080):
        self._camera = Picamera2()

        config = self._camera.create_video_configuration(
            main={
                "size": (width, height),
                "format": "BGR888",
            }
        )

        self._camera.configure(config)
        self._camera.start()

        # Let libcamera finish sensor startup.
        time.sleep(1.0)

    def read(self):
        try:
            return self._camera.capture_array()
        except Exception:
            return None

    def close(self):
        if self._camera is not None:
            self._camera.stop()
            self._camera.close()
            self._camera = None
