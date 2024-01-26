from dolfinx.io import XDMFFile

class Output:
    def _create_output(self,filename,**kwargs):
        self.outfile = XDMFFile(self.mesh.comm, filename, "w",**kwargs)
        self.outfile.write_mesh(self.mesh)
    
    def write_output(self):
        # The error 'int object is not iterable' only appears
        # with hexahedral meshes
        for fun in self.solution.next.split():
            self.outfile.write_function(u=fun, t=self.time)