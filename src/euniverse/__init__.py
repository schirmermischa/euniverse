import os

def get_resource(filename):
    """Get the absolute path to a resource file within the package."""
    return os.path.join(os.path.dirname(__file__), filename)
