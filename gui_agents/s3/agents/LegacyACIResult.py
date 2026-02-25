from dataclasses import dataclass


@dataclass
class LegacyACIResult:
    result: str
    feedback_image_bytes: bytes
    annotation: str


    @property
    def feedback_image_base64(self) -> str:
        import base64
        return base64.b64encode(self.feedback_image_bytes).decode()