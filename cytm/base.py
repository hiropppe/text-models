from .util import save_model
from .util import load_model


class BaseModel:

    @classmethod
    def load(cls, model_path):
        return load_model(model_path)

    def save(self, output_path):
        save_model(self, output_path)

