from dolfinx.io import XDMFFile

class Output:
    def _create_output(self,filename,**kwargs):
        self.outfile = XDMFFile(self.mesh.comm, filename,**kwargs)
    
    def write_output(self):
        # The error 'int object is not iterable' only appears
        # with hexahedral meshes
        self.outfile.write_function(*[fun for fun in self.solution.next.subfunctions], time=self.time)