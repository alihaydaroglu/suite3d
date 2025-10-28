import os
from functools import wraps
from warnings import warn

suite3d_developer_mode = os.environ.get("SUITE3D_DEVELOPER", "").lower() == "true"

def todo(message):
    """
    A function to call that prints warning-like messages wherever work needs to be done on a function. 

    Args:
        message (str): The message to print.
    """
    if suite3d_developer_mode:
        warn(message, stacklevel=2)

def deprecated(reason=None):
    """
    A decorator to mark functions as deprecated as we refactor the code. It's kind of overkill, but 
    it's nice because then we can just search for "deprecated" in the library to find any functions
    that need to be updated or removed. 
    
    Args:
        reason (Optional[str]) : An explanation for why the function is deprecated and/or what to use instead.
    """
    def decorator(func):
        message = f"Function {func.__name__} is deprecated."
        if reason:
            message += f" {reason}"
        @wraps(func)
        def wrapper(*args, **kwargs):
            warn(message, category=DeprecationWarning, stacklevel=3)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def deprecated_inputs(*explanation):
    """
    A decorator to mark functions with deprecated inputs as we refactor the code. Like the above decorator,
    we can search for "deprecated_inputs" in the library to find any functions that need their inputs to be
    reviewed and checked. 

    Args:
        *explanation (str): One or more strings providing explanations for which inputs are deprecated.
                            At least one explanation must be provided.
    """
    assert len(explanation) > 0, "You must provide an explanation for the deprecated inputs."
    def decorator(func):
        message = f"Function {func.__name__} is using deprecated inputs."
        for exp in explanation:
            message += "\nDeprecated_inputs: " + exp
        @wraps(func)
        def wrapper(*args, **kwargs):
            if suite3d_developer_mode:
                warn(message, category=DeprecationWarning, stacklevel=3)
            return func(*args, **kwargs)
        return wrapper
    return decorator
