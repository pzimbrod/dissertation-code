from firedrake import File

class Output:
    def _create_output(self,filename,**kwargs):
        self.outfile = File(filename,**kwargs)
    
    def write_output(self):
        # The error 'int object is not iterable' only appears
        # with hexahedral meshes
        self.outfile.write(*[fun for fun in self.solution.next.subfunctions], time=self.time)