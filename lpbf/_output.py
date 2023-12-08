from firedrake import File

class Output:
    def _create_output(self,filename="lpbf",project_output=True):
        self.outfile = File(filename=f"{filename}.pvd",project_output=project_output)
    
    def write_output(self):
        None