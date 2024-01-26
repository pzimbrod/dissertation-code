from dolfinx.fem import Function

class TimeDependentFunction:
    """
    A basic abstraction for a Firedrake function that is time dependent.
    It only has two Functions as attributes: the previous value (t) and
    the next one (t+1).
    """
    def __init__(self,previous: Function, next: Function) -> None:
        self.previous = previous
        self.next     = next