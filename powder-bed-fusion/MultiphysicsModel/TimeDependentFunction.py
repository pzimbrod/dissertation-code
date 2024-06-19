from dolfinx.fem import Function

class TimeDependentFunction:
    """
    A basic abstraction for a FEniCS function that is time dependent.
    It only has two Functions as attributes: the previous value (t) and
    the next one (t+1).
    """
    def __init__(self,previous: Function, current: Function) -> None:
        self.previous = previous
        self.current  = current

    def update(self) -> None:
        """
        Do a timestep update, i.e. assign the values of the current
        function to the previous one
        """
        self.previous.x.array[:] = self.current.x.array

        self.previous.x.scatter_forward()
        self.current.x.scatter_forward()