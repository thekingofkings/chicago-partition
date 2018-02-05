import MCMC

# Global variables
project_name = 'case-study-crime'
targetName = 'total'


# Initialize MCMC: learn regression using administrative boundaries
MCMC.initialize(project_name=project_name, targetName=targetName, lmbd=0.75, f_sd=1.5, Tt=10)

