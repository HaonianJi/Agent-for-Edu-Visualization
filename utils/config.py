import importlib

def name_to_class(cfg):
    module = importlib.import_module(cfg.module_name)
    moduleclass = getattr(module, cfg.class_name)
    class_ = moduleclass(cfg)
    return class_