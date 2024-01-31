from dolfinx.io import XDMFFile

class Output:
    def _create_output(self,filename,**kwargs):
        self.outfile = XDMFFile(self.mesh.comm, filename, "w",**kwargs)
        with self.outfile as file:
            file.write_mesh(self.mesh)
    
    def write_output(self):
        # The error 'int object is not iterable' only appears
        # with hexahedral meshes
        with self.outfile as file:
            for fun in self.solution.next.split():
                file.write_function(u=fun, t=self.time)