'''
A PBD simulator for mass-spring systems

Renderer: SAPIEN renderer
'''

# ffmpeg -framerate 50 -i ../output/frames/step_%04d.png -vf "fps=50,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" ../output/video/out.gif

import sapien
import numpy as np
import os
import igl
from PIL import Image
import time
import warp as wp

wp.init()
device = wp.get_preferred_device()

sapien.render.set_viewer_shader_dir("rt")
sapien.render.set_camera_shader_dir("rt")
sapien.render.set_ray_tracing_samples_per_pixel(8)
sapien.render.set_ray_tracing_denoiser("oidn")


def get_spring_constraints(vertices, edges, spring_stiffness, constraints):
    for edge in edges:
        constraint = {
            "type": "spring",
            "indices": edge,
            "stiffness": spring_stiffness,
            "rest_length": np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]),
        }
        constraints.append(constraint)


def get_mask(vertices):
    n_particles = vertices.shape[0]
    mask = np.ones(n_particles, dtype=np.float32)
    x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])
    for i in range(n_particles):
        x, y, z = vertices[i]
        if n_particles == 2:
            if x == x_min and y == y_min and z == z_max:
                mask[i] = 0.0
        else:
            if x == x_min and (y == y_min or y == y_max):
                mask[i] = 0.0
    return mask


def init_renderer(scene: sapien.Scene, vertices, faces):
    ######## Simple renderer ########
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([1, 0, -1], [1, 1, 1], True)
    scene.add_ground(0.0)

    # add a camera to indicate shader
    cam_entity = sapien.Entity()
    cam = sapien.render.RenderCameraComponent(512, 512)
    cam.set_near(1e-3)
    cam.set_far(100)
    cam_entity.add_component(cam)
    cam_entity.name = "camera"
    cam_entity.set_pose(
        # sapien.Pose([-0., -0.8, 2.7], [0.73446, -0.146693, 0.1681, 0.64093])
        sapien.Pose([-0.133168, -0.777414, 2.709], [0.73446, -0.146693, 0.1681, 0.64093])
    )
    scene.add_entity(cam_entity)

    cloth_entity = sapien.Entity()
    cloth_components = sapien.render.RenderCudaMeshComponent(len(vertices), 2 * len(faces))
    cloth_components.set_vertex_count(len(vertices))
    cloth_components.set_triangle_count(2 * len(faces))
    cloth_components.set_triangles(np.concatenate([faces, faces[:, ::-1]], axis=0))
    cloth_components.set_material(sapien.render.RenderMaterial(base_color=[0.7, 0.3, 0.4, 1.0]))
    cloth_entity.add_component(cloth_components)
    scene.add_entity(cloth_entity)

    vertex_components = [sapien.Entity() for _ in range(len(vertices))]
    for i in range(len(vertices)):
        vertex_components[i].set_pose(sapien.Pose(vertices[i]))
        render_shape_sphere = sapien.render.RenderShapeSphere(
            1e-2,
            sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.7, 0.8]),
        )
        render_component = sapien.render.RenderBodyComponent()
        render_component.attach(render_shape_sphere)
        vertex_components[i].add_component(render_component)
    for i in range(len(vertices)):
        scene.add_entity(vertex_components[i])

    return cam, cloth_components, vertex_components


@wp.kernel
def copy_positions_to_render(
    dst_vertices: wp.array2d(dtype=wp.float32),
    src_positions: wp.array(dtype=wp.vec3),
):
    i, j = wp.tid()
    dst_vertices[i, j] = src_positions[i][j]


def update_render_component(render_component: sapien.render.RenderCudaMeshComponent, vertex_components, x):
    x_wp = wp.array(x, dtype=wp.vec3, device=device)
    interface = render_component.cuda_vertices.__cuda_array_interface__
    dst = wp.array(
        ptr=interface["data"][0],
        dtype=wp.float32,
        shape=interface["shape"],
        strides=interface["strides"],
        owner=False,
        device=device,
    )
    wp.launch(
        kernel=copy_positions_to_render,
        dim=(len(x), 3),
        inputs=[dst, x_wp],
        device=device,
    )
    render_component.notify_vertex_updated(wp.get_stream(device).cuda_stream)

    for i in range(len(vertex_components)):
        vertex_components[i].set_pose(sapien.Pose(x[i]))


def spring_C(x0, x1, l):
    return np.linalg.norm(x1 - x0) - l


def spring_C_grad(x0, x1, l):
    n = x1 - x0
    n = n / np.linalg.norm(n)
    return -n, n


def pbd_step(x, mask, la, masses, constraints, time_step):
    for j, c in enumerate(constraints):
        if c["type"] == "spring":
            i0, i1 = c["indices"]
            l = c["rest_length"]
            k = c["stiffness"]

            x0, x1 = x[i0], x[i1]
            m0, m1 = masses[i0], masses[i1]
            
            C = spring_C(x0, x1, l) 
            dC_dx0, dC_dx1 = spring_C_grad(x0, x1, l)  # shape: (3,)
            alpha = 1.0 / (k * time_step * time_step)

            la_upd = (-C - alpha * la[j]) / (1.0 / m0 + 1.0 / m1 + alpha)
            # print(f"la_upd: {la_upd}, C: {C}, alpha: {alpha}")
            la[j] += la_upd
            x[i0] += la_upd * dC_dx0 / m0 * mask[i0]
            x[i1] += la_upd * dC_dx1 / m1 * mask[i1]

            # f = -k * C * dC_dx1
            # print(f"x: {x0, x1}, f: {f}")
            # print(f"la: {la[j]}")
        else:
            raise NotImplementedError(f"Constraint type {c['type']} not implemented")


def sim_step(x, x_prev, v, mask, gravity, la, masses, constraints, time_step, n_pbd_iters):
    x_prev[...] = x
    x += (time_step * v + (time_step * time_step) * gravity) * mask[:, None]
    la *= 0.0

    for pbd_iter in range(n_pbd_iters):
        pbd_step(x, mask, la, masses, constraints, time_step)

    v[...] = (x - x_prev) / time_step


def main():
    ######## Hyperparameters ########
    # solver settings
    time_step = 0.01
    n_pbd_iters = 10

    # scene settings
    gravity = np.array([0, 0, -9.8])
    spring_stiffness = 1e3
    density = 1e3
    thickness = 1e-3
    init_pos = np.array([0.0, 0.0, 2.5])

    # rendering and exporting settings
    n_render_steps = int(1e9)
    n_render_steps = 200
    render_every = 2  # Render every "render_every" time steps
    save_render = False
    save_render_dir = os.path.join(os.path.dirname(__file__), '../output/frames')
    if save_render:
        os.makedirs(save_render_dir, exist_ok=True)

    ######## Load geometry ########
    assets_dir = os.path.join(os.path.dirname(__file__), '../assets')
    vertices, faces = igl.read_triangle_mesh(os.path.join(assets_dir, "cloth_11.obj"))
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)
    edges = igl.edges(faces)
    print(f"vertices: {vertices.shape}, faces: {faces.shape}, edges: {edges.shape}")

    # vertices = np.array([
    #     [0, 0, 0],
    #     [0, 0, -1],
    # ], dtype=np.float32)
    # faces = np.array([
    #     [0, 1, 0],
    # ], dtype=np.int32)
    # edges = np.array([
    #     [0, 1],
    # ], dtype=np.int32)
    
    ######## Create mass spring system ########
    n_particles = vertices.shape[0]
    masses = np.ones(n_particles, dtype=np.float32) * density * thickness / n_particles
    # print(masses)
    constraints = []
    get_spring_constraints(vertices, edges, spring_stiffness, constraints)
    # print(constraints)

    ######## Initialize simulation ########
    x = vertices.copy() + init_pos
    v = np.zeros_like(x)
    x_prev = x.copy()
    mask = get_mask(x)
    la = np.zeros(len(constraints), dtype=np.float32)

    ######## Initialize renderer ########
    scene = sapien.Scene()
    cam, cloth_components, vertex_components = init_renderer(scene, vertices, faces)

    viewer = sapien.utils.Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_pose(cam.entity_pose)
    viewer.window.set_camera_parameters(1e-3, 1000, np.pi / 2)

    viewer.paused = True
    update_render_component(cloth_components, vertex_components, x)
    scene.update_render()
    viewer.render()
    
    ######## Simulation loop ########
    for i in range(n_render_steps):
        for j in range(render_every):
            sim_step(x, x_prev, v, mask, gravity, la, masses, constraints, time_step, n_pbd_iters)
        update_render_component(cloth_components, vertex_components, x)
        scene.update_render()
        viewer.render()

        # print(x)

        if save_render:
            cam.take_picture()
            rgba = cam.get_picture("Color")
            rgba = np.clip(rgba, 0, 1)[:, :, :3]
            rgba = Image.fromarray((rgba * 255).astype(np.uint8))
            rgba.save(os.path.join(save_render_dir, f"step_{i:04d}.png"))

main()
