import maya.cmds as cmds
import random


# The problem with this is that 'nodeType' returns the
# most derived type of the node. In this example, "surfaceShape"
# is a base type for nurbsSurface so nothing will be printed.
# To do this properly, the -typ/type flag should be used
# to list objects of a specific type as in:

# Comanda pentru afisat toate componentele dintr-o figura
allObjects = cmds.ls(type='surfaceShape')
for obj in allObjects:
    print obj


print('LSEdddD',random.randrange(0,100,2))
