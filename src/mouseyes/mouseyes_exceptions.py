class ModelXmlFileNotFoundException(Exception):
    """When there is no model XML file in the specified path"""
    pass

class ModelBinFileNotFoundException(Exception):
    """When there is no model BIN file in the specified path"""
    pass

class UnsupportedLayersException(Exception):
    """There are unsupported layers in the model"""
    pass