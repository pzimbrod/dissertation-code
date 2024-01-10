from firedrake import FunctionSpace, Function

class TimeDependentFunction:
    """
    A basic abstraction for a Firedrake function that is time dependent.
    It only has two Functions as attributes: the previous value (t) and
    the next one (t+1).
    """
    def __init__(self,fs: FunctionSpace, name: str) -> None:
        fn_previous = Function(fs, name=f"{name}_previous")
        fn_next = Function(fs, name=f"{name}_current")
        self.previous = fn_previous
        self.next     = fn_next