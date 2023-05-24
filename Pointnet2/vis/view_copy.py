import open3d as o3d
import numpy as np

def mini_color_table(index, diff, norm=True):
    colors = [
        [0.5000, 0.2500, 0.0000], [0.2000, 0.8000, 0.2000], [0.9900, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.5000], [0.9900, 0.0000, 0.9900], [0.1000, 0.1000, 0.1000],
        [0.9900, 0.9900, 0.0000], [0.5000, 0.4000, 0.9900], [0.0000, 0.0000, 0.0000],
    ]

    if diff:
        colors = [
            [0.0,0.9,0], [0.9,0.0,0],  [0.99,0,0],
            [0.5,0.5,0.5], [0.99,0,0.99], [0.6,0.4,0.6],
            [0.99,0.99,0], [0.15,0.09,0.9], [0.0000, 0.0000, 0.0000],
        ]

    else:
        colors = [
            [0.5,0.25,0], [0.1,0.6,0], [0.99,0,0],
            [0.5,0.5,0.5], [0.99,0,0.99], [0.6,0.4,0.6],
            [0.99,0.99,0], [0.15,0.09,0.9], [0.0000, 0.0000, 0.0000],
        ]

    assert index >= 0 and index < len(colors)
    color = colors[index]

    if not norm:
        color[0] *= 255
        color[1] *= 255
        color[2] *= 255

    return color

def pinta_mis_colores(labels):
    """
    MisColores = {
        1: np.array([0.5,0.25,0]), #terreno
        2: np.array([0.1,0.6,0]),  #vegetación
        3: np.array([0.99,0,0]), #coche
        4: np.array([0.5,0.5,0.5]), #torre
        5: np.array([0.99,0,0.99]), #cable
        6: np.array([0.1,0.1,0.1]), #valla/muro
        7: np.array([0.99,0.99,0]), #farola
        8: np.array([0.5,0.9,0.09]), #edificio
        21: np.array([0,0,0]) #ignorar
        }
    """
    MisColores = {
    0: np.array([0.5,0.25,0]), #terreno
    1: np.array([0.1,0.6,0]),  #vegetación
    2: np.array([0.99,0,0]), #coche
    3: np.array([0.5,0.5,0.5]), #torre
    4: np.array([0.99,0,0.99]), #cable
    5: np.array([0.6,0.4,0.6]), #valla/muro
    6: np.array([0.99,0.99,0]), #farola
    7: np.array([0.15,0.09,0.9]), #edificio
    }

    Colorin = [None] * len(labels)
    for x in range(len(labels)):
        for col in MisColores:
            if (int(col) == int(labels[x])):
                Colorin[x]=MisColores[col]

    return Colorin

def view_points(points, colors=None, gt_colors=None ):
    '''
    points: np.ndarray with shape (n, 3)
    colors: [r, g, b] or np.array with shape (n, 3)
    '''
    #Estaría interesante crear un toggle de visualizacion con el gt y el color
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points) 

    if colors is not None:
        if isinstance(colors, np.ndarray):
            cloud.colors = o3d.utility.Vector3dVector(colors)
        else: cloud.paint_uniform_color(colors)

    o3d.visualization.draw_geometries([cloud])
    




def label2color(labels, diff):
    '''
    labels: np.ndarray with shape (n, )
    colors(return): np.ndarray with shape (n, 3)
    '''
    num = labels.shape[0]
    colors = np.zeros((num, 3))

    minl, maxl = np.min(labels), np.max(labels)
    for l in range(minl, maxl + 1):
        colors[labels==l, :] = mini_color_table(l,diff) 

    return colors


def view_points_labels(points, labels, groundtruth, diff=False):
    '''
    Assign points with colors by labels and view colored points.
    points: np.ndarray with shape (n, 3)
    labels: np.ndarray with shape (n, 1), dtype=np.int32
    '''

    print("-------")
    print("-------")
    import time
    time.sleep(5)
    print(points)
    print(points.shape)
    print("------")
    print(labels)
    print(labels.shape)

    time.sleep(599)
    assert points.shape[0] == labels.shape[0]
    assert points.shape[0] == groundtruth.shape[0]

    ground_truth_colors = label2color(groundtruth, diff)
    colors = label2color(labels, diff)
    #mis_colores = pinta_mis_colores(labels)

    view_points(points, colors, ground_truth_colors)

#se le ha puesto para mostrar las diff, pasando un param mas que es diff, y tambien gorund trurth es nuevo