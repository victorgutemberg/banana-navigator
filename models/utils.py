import importlib

def get_model(model_name):
    return getattr(importlib.import_module(f'models.{model_name}'), 'QNetwork')