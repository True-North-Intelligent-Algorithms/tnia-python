from helper import make_3D_random_spheres
from tnia.plotting.projections import show_xy_zy_max
from matplotlib.figure import Figure

def test_projections():

    random_spheres = make_3D_random_spheres()

    print()
    print('sum random spheres:',random_spheres.sum())
    # print num elements
    print('num elelements random spheres:',random_spheres.size)
    assert random_spheres.sum() == 8272.0

    fig = show_xy_zy_max(random_spheres)

    print(type(fig))

    assert isinstance(fig, Figure)