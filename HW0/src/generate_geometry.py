import igl
import os
import numpy as np
import scipy.sparse as sp


cloth_res = (2, 2)
cloth_shape = (1.0, 1.0)

grid_vertices_2d = q_rest_arr2d = np.stack(
    np.meshgrid(
        np.linspace(0, cloth_shape[0], cloth_res[0], dtype=np.float32),
        np.linspace(0, cloth_shape[1], cloth_res[1], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    ),
    axis=-1,
)
grid_vertices_id_2d = np.arange(cloth_res[0] * cloth_res[1], dtype=np.int32).reshape(
    cloth_res
)

# Add one vertex at the center of each square cell
cell_shape = (cloth_shape[0] / (cloth_res[0] - 1), cloth_shape[1] / (cloth_res[1] - 1), 0.)
cell_centers_2d = (
    grid_vertices_2d[:-1, :-1] + np.array(cell_shape, dtype=np.float32) / 2.0
)
cell_centers_id_2d = (
    np.arange((cloth_res[0] - 1) * (cloth_res[1] - 1), dtype=np.int32).reshape(
        (cloth_res[0] - 1, cloth_res[1] - 1)
    )
    + cloth_res[0] * cloth_res[1]
)

# complete vertices
vertices = np.concatenate(
    [
        grid_vertices_2d.reshape(-1, 3),
        cell_centers_2d.reshape(-1, 3),
    ],
    axis=0,
)

# Create 4 triangles for each square cell
#
# g[i, j] ---  g[i, j+1]
#    | \   0   /  |
#    |1  c[i, j] 3|
#    | /   2   \  |
# g[i+1, j] -- g[i+1, j+1]
#
faces = []
for i in range(cloth_res[0] - 1):
    for j in range(cloth_res[1] - 1):
        faces.append(
            (
                grid_vertices_id_2d[i, j],
                grid_vertices_id_2d[i, j + 1],
                cell_centers_id_2d[i, j],
            )
        )
        faces.append(
            (
                grid_vertices_id_2d[i, j],
                cell_centers_id_2d[i, j],
                grid_vertices_id_2d[i + 1, j],
            )
        )
        faces.append(
            (
                grid_vertices_id_2d[i + 1, j],
                cell_centers_id_2d[i, j],
                grid_vertices_id_2d[i + 1, j + 1],
            )
        )
        faces.append(
            (
                cell_centers_id_2d[i, j],
                grid_vertices_id_2d[i, j + 1],
                grid_vertices_id_2d[i + 1, j + 1],
            )
        )
faces = np.array(faces, dtype=np.int32)

output_dir = os.path.join(os.path.dirname(__file__), '../assets')
os.makedirs(output_dir, exist_ok=True)
igl.write_triangle_mesh(os.path.join(output_dir, f"cloth_{cloth_res[0]}.obj"), vertices, faces)
