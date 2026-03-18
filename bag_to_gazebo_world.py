#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bag_to_gazebo_world.py

Unified ROS 2 Bag -> Registered Point Cloud -> Poisson Mesh -> Gazebo World
Extracts a point cloud from a ROS 2 bag, registers it, generates a mesh,
and exports it as a Gazebo simulation environment (.world, .sdf, .config, .stl).
"""

import argparse
import bisect
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ----------------------------
# Gazebo Templates
# ----------------------------
CONFIG_TEMPLATE = """<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>bag_to_gazebo_world</name>
    <email>auto@generated.com</email>
  </author>
  <description>
    3D environment mesh generated from a ROS 2 bag file.
  </description>
</model>
"""

SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <mesh>
            <uri>model://{model_name}/meshes/model.stl</uri>
          </mesh>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>model://{model_name}/meshes/model.stl</uri>
          </mesh>
        </geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>{gazebo_material}</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

WORLD_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">
    <include>
      <uri>model://sun</uri>
    </include>
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://{model_name}</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>
  </world>
</sdf>
"""

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)

POINTFIELD_TO_DTYPE = {
    7: np.float32,  # FLOAT32
    8: np.float64,  # FLOAT64
}


def convert_ros_pc2_to_o3d(msg):
    try:
        fields = {f.name: (int(f.offset), int(f.datatype)) for f in msg.fields}
        if "x" not in fields or "y" not in fields or "z" not in fields:
            return None

        xoff, xdt = fields["x"]
        yoff, ydt = fields["y"]
        zoff, zdt = fields["z"]

        if xdt not in POINTFIELD_TO_DTYPE or ydt not in POINTFIELD_TO_DTYPE or zdt not in POINTFIELD_TO_DTYPE:
            return None

        if not (xdt == ydt == zdt):
            return None

        npdt = POINTFIELD_TO_DTYPE[xdt]
        npoints = int(msg.width) * int(msg.height)
        if npoints <= 0:
            return None

        itemsize = int(msg.point_step)
        if itemsize <= 0:
            return None

        dtype = np.dtype({
            "names": ["x", "y", "z"],
            "formats": [npdt, npdt, npdt],
            "offsets": [xoff, yoff, zoff],
            "itemsize": itemsize,
        })

        arr = np.frombuffer(msg.data, dtype=dtype, count=npoints)
        pts = np.empty((npoints, 3), dtype=np.float64)
        pts[:, 0] = arr["x"]
        pts[:, 1] = arr["y"]
        pts[:, 2] = arr["z"]

        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        if pts.shape[0] < 10:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    except Exception:
        return None


def get_odom_transform(odom_msg):
    try:
        pos = odom_msg.pose.pose.position
        quat = odom_msg.pose.pose.orientation
        t = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        rot = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = t
        return T
    except Exception:
        return None


def get_closest_timestamp(ts, sorted_keys: list):
    if not sorted_keys:
        return None
    idx = bisect.bisect_left(sorted_keys, ts)
    if idx == 0:
        return sorted_keys[0]
    if idx == len(sorted_keys):
        return sorted_keys[-1]
    before, after = sorted_keys[idx - 1], sorted_keys[idx]
    return before if (ts - before) <= (after - ts) else after


def compute_fpfh_descriptor(pcd, voxel_size: float):
    radius_normal = voxel_size * 2.0
    radius_feature = voxel_size * 5.0

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return fpfh


def ransac_coarse_alignment(source, target, source_fpfh, target_fpfh, voxel_size: float, ransac_thresh_mult: float = 5.0):
    distance_threshold = voxel_size * float(ransac_thresh_mult)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000, 0.999),
    )

    if result.fitness > 0.1:
        return result.transformation
    return None


def detect_loop_closure(
    current_idx: int, current_pcd, current_fpfh, historical_pcds, historical_fpfhs,
    historical_poses, voxel_size: float, search_radius: float = 10.0,
    loop_fitness_thresh: float = 0.3, temporal_window: int = 100,
):
    if current_idx < temporal_window:
        return []

    search_indices = list(range(0, current_idx - temporal_window))
    if not search_indices:
        return []

    current_pos = historical_poses[current_idx][:3, 3]
    hist_positions = np.array([historical_poses[i][:3, 3] for i in search_indices], dtype=np.float64)
    if hist_positions.shape[0] == 0:
        return []

    pos_pcd = o3d.geometry.PointCloud()
    pos_pcd.points = o3d.utility.Vector3dVector(hist_positions)
    kdtree = o3d.geometry.KDTreeFlann(pos_pcd)
    _k, idxs, _dist2 = kdtree.search_radius_vector_3d(current_pos, float(search_radius))
    if not idxs:
        return []

    candidate_indices = [search_indices[i] for i in idxs]
    loop_closures = []

    for cand_idx in candidate_indices:
        cand_fpfh = historical_fpfhs[cand_idx]
        if cand_fpfh is None:
            continue

        cand_pcd = copy.deepcopy(historical_pcds[cand_idx])
        coarse = ransac_coarse_alignment(current_pcd, cand_pcd, current_fpfh, cand_fpfh, voxel_size)
        if coarse is None:
            continue

        icp = o3d.pipelines.registration.registration_icp(
            current_pcd, cand_pcd, voxel_size * 2.0, coarse,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
        )

        if float(icp.fitness) >= float(loop_fitness_thresh):
            loop_closures.append((cand_idx, icp.transformation, float(icp.fitness)))

    return loop_closures


def _safe_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, eps, None)


def attach_view_rays_as_normals(pcd_world: o3d.geometry.PointCloud, sensor_origin_world: np.ndarray) -> None:
    pts = np.asarray(pcd_world.points, dtype=np.float64)
    if pts.shape[0] == 0:
        return
    ray_dirs = sensor_origin_world.reshape(1, 3) - pts
    ray_dirs = _safe_normalize(ray_dirs, eps=1e-6)
    pcd_world.normals = o3d.utility.Vector3dVector(ray_dirs)


def orient_geometric_normals_with_view_rays(pcd: o3d.geometry.PointCloud, view_rays: np.ndarray) -> None:
    geom = np.asarray(pcd.normals, dtype=np.float64)
    if geom.shape[0] == 0:
        return
    vr = np.asarray(view_rays, dtype=np.float64)
    if vr.shape != geom.shape:
        return

    vr = _safe_normalize(vr, eps=1e-6)
    geom = _safe_normalize(geom, eps=1e-12)
    dots = np.sum(geom * vr, axis=1)
    geom[dots < 0.0] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(geom)


def estimate_geometric_normals_oriented(
    pcd: o3d.geometry.PointCloud, voxel_size: float, view_rays: np.ndarray | None,
) -> None:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3.0, max_nn=30)
    )
    pcd.normalize_normals()

    if view_rays is not None and view_rays.shape[0] == len(pcd.points):
        orient_geometric_normals_with_view_rays(pcd, view_rays)
    else:
        try:
            pcd.orient_normals_consistent_tangent_plane(100)
        except Exception:
            pass


def clean_point_cloud(
    pcd: o3d.geometry.PointCloud, voxel_size: float, do_voxel_downsample: bool = True,
) -> o3d.geometry.PointCloud:
    pcd_clean = pcd

    if do_voxel_downsample:
        logging.info("Voxel downsampling...")
        pcd_clean = pcd_clean.voxel_down_sample(voxel_size)

    logging.info("Radius outlier removal...")
    try:
        pcd_tmp, _ = pcd_clean.remove_radius_outlier(nb_points=12, radius=voxel_size * 3.0)
        if len(pcd_tmp.points) > 0:
            pcd_clean = pcd_tmp
        else:
            logging.warning("ROR produced empty cloud; skipping.")
    except Exception as e:
        logging.warning(f"ROR failed ({e}); skipping.")

    logging.info("Statistical outlier removal...")
    try:
        pcd_tmp, _ = pcd_clean.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(pcd_tmp.points) > 0:
            pcd_clean = pcd_tmp
        else:
            logging.warning("SOR produced empty cloud; skipping.")
    except Exception as e:
        logging.warning(f"SOR failed ({e}); skipping.")

    logging.info("DBSCAN clustering (keeping largest component)...")
    try:
        labels = np.array(
            pcd_clean.cluster_dbscan(eps=voxel_size * 4.0, min_points=30, print_progress=False)
        )
        if len(labels) > 0 and labels.max() >= 0:
            largest_cluster_idx = np.bincount(labels[labels >= 0]).argmax()
            pcd_tmp = pcd_clean.select_by_index(np.where(labels == largest_cluster_idx)[0])
            if len(pcd_tmp.points) > 0:
                pcd_clean = pcd_tmp
            else:
                logging.warning("DBSCAN isolated 0 points; skipping.")
        else:
            logging.warning("DBSCAN found no valid clusters; skipping.")
    except Exception as e:
        logging.warning(f"DBSCAN failed ({e}); skipping.")

    return pcd_clean


def create_mesh(
    pcd: o3d.geometry.PointCloud,
    depth: int = 9,
    min_density_percentile: float = 1.0,
    max_vertex_distance: float = 0.15,
    workers: int = 4,
    decimate_target: float | None = None,
):
    logging.info(f"Input point cloud: {len(pcd.points)} points")

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        try:
            pcd.orient_normals_consistent_tangent_plane(100)
        except Exception:
            pass

    logging.info(f"Running Poisson Reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth), linear_fit=True
    )
    logging.info(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    densities = np.asarray(densities, dtype=np.float64)
    if densities.size > 0:
        thr = np.percentile(densities, float(min_density_percentile))
        logging.info(f"Density trim at bottom {min_density_percentile}% (threshold={thr:.4f})")
        mesh.remove_vertices_by_mask(np.asarray(densities < thr, dtype=bool))

    if len(mesh.triangles) == 0:
        raise ValueError("Mesh is empty after density trim. Try a lower --min_density_percentile.")

    p = np.asarray(pcd.points)
    v = np.asarray(mesh.vertices)

    if len(p) > 0 and len(v) > 0 and max_vertex_distance and float(max_vertex_distance) > 0:
        logging.info(f"Distance trim: removing vertices farther than {max_vertex_distance}m...")
        tree = cKDTree(p)
        d, _ = tree.query(v, k=1, workers=workers)
        mesh.remove_vertices_by_mask(np.asarray(d > float(max_vertex_distance), dtype=bool))

    logging.info("Cleaning mesh structure...")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if len(mesh.triangles) == 0:
        raise ValueError("Mesh is empty after distance trim. Try a larger --max_vertex_distance.")

    if decimate_target is not None:
        n_before = len(mesh.triangles)
        target_tris = max(1, int(n_before * decimate_target)) if decimate_target <= 1.0 else int(decimate_target)
        logging.info(f"Decimating mesh: {n_before} -> ~{target_tris} triangles...")
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_tris, boundary_weight=10.0
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        logging.info(f"Decimated mesh: {len(mesh.triangles)} triangles")

    logging.info(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    return mesh


def process_bag(args):
    bag_path = Path(args.bagpath)
    out_dir = Path(args.outputdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        sys.exit(f"Error: Bag file not found: {bag_path}")

    model_name = args.model_name
    pc_topic = args.pc_topic
    odom_topic = args.odom_topic

    logging.info(f"Reading bag: {bag_path}")
    logging.info(f"Output model name: {model_name}")

    point_clouds = []
    odom_data = {}

    topics_to_read = [pc_topic]
    if odom_topic:
        topics_to_read.append(odom_topic)

    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        conns = [c for c in reader.connections if c.topic in topics_to_read]
        if not conns:
            sys.exit(f"Error: No messages found for topics: {topics_to_read}")

        for conn, ts, raw in tqdm(reader.messages(connections=conns), desc="Reading messages"):
            try:
                msg = reader.deserialize(raw, conn.msgtype)
                if conn.topic == pc_topic:
                    pcd = convert_ros_pc2_to_o3d(msg)
                    if pcd is not None and len(pcd.points) >= 100:
                        point_clouds.append((ts, pcd))
                elif odom_topic and conn.topic == odom_topic:
                    T = get_odom_transform(msg)
                    if T is not None:
                        odom_data[ts] = T
            except Exception:
                continue

    if not point_clouds:
        sys.exit("Error: No valid point clouds were extracted.")

    logging.info(f"Extracted {len(point_clouds)} point clouds")
    if odom_topic:
        if len(odom_data) == 0:
            logging.warning("Odometry topic specified but no messages found; falling back to identity initial guess.")
        else:
            logging.info(f"Extracted {len(odom_data)} odometry messages")

    odom_ts_sorted = sorted(odom_data.keys())
    odom_max_latency_ns = int(args.odom_max_latency * 1e9)

    pose_graph = o3d.pipelines.registration.PoseGraph()
    current_transform = np.eye(4, dtype=np.float64)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(current_transform.copy()))

    _ts0, src_raw = point_clouds[0]
    src = src_raw.voxel_down_sample(args.voxel_size)
    src.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=30)
    )

    if args.enable_loop_closure:
        src_fpfh = compute_fpfh_descriptor(src, args.voxel_size)
        accumulated_pcds = [src]
        accumulated_fpfhs = [src_fpfh]
        accumulated_poses = [current_transform.copy()]

    previous_odom_T = None
    if odom_topic and odom_ts_sorted:
        closest_ts = get_closest_timestamp(_ts0, odom_ts_sorted)
        if closest_ts is not None and abs(closest_ts - _ts0) < odom_max_latency_ns:
            previous_odom_T = odom_data[closest_ts]

    successful_pc_indices = [0]
    logging.info("Registering point clouds...")

    for i in tqdm(range(1, len(point_clouds)), desc="Registering"):
        ts, tgt_raw = point_clouds[i]
        tgt = tgt_raw.voxel_down_sample(args.voxel_size)
        tgt.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=30)
        )

        initial_guess = np.eye(4, dtype=np.float64)
        if odom_topic and odom_ts_sorted:
            closest_ts = get_closest_timestamp(ts, odom_ts_sorted)
            if closest_ts is not None and abs(closest_ts - ts) < odom_max_latency_ns:
                current_odom_T = odom_data[closest_ts]
                if previous_odom_T is not None:
                    initial_guess = np.linalg.inv(previous_odom_T) @ current_odom_T
                previous_odom_T = current_odom_T
            else:
                previous_odom_T = None
                initial_guess = np.eye(4)

        try:
            reg = o3d.pipelines.registration.registration_icp(
                src, tgt, args.icp_dist_thresh, initial_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
            )
        except Exception:
            continue

        if float(reg.fitness) < float(args.icp_fitness_thresh):
            continue

        current_transform = reg.transformation @ current_transform
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(current_transform))
        )

        info = np.eye(6, dtype=np.float64) * max(float(reg.fitness), 1e-6)
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                len(pose_graph.nodes) - 2, len(pose_graph.nodes) - 1,
                reg.transformation, info, uncertain=False,
            )
        )

        if args.enable_loop_closure:
            accumulated_pcds.append(tgt)
            accumulated_poses.append(current_transform.copy())
            do_lc = (i % args.loop_closure_search_interval == 0)
            tgt_fpfh = compute_fpfh_descriptor(tgt, args.voxel_size) if do_lc else None
            accumulated_fpfhs.append(tgt_fpfh)

            if do_lc:
                lcs = detect_loop_closure(
                    current_idx=len(accumulated_pcds) - 1,
                    current_pcd=tgt,
                    current_fpfh=tgt_fpfh,
                    historical_pcds=accumulated_pcds,
                    historical_fpfhs=accumulated_fpfhs,
                    historical_poses=accumulated_poses,
                    voxel_size=args.voxel_size,
                    search_radius=args.loop_closure_radius,
                    loop_fitness_thresh=args.loop_closure_fitness_thresh,
                )
                for cand_idx, lc_T, lc_fit in lcs:
                    lc_info = np.eye(6, dtype=np.float64) * (max(float(lc_fit), 1e-6) * 100.0)
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            cand_idx, len(pose_graph.nodes) - 1, lc_T, lc_info, uncertain=True,
                        )
                    )

        successful_pc_indices.append(i)
        src = tgt

    if len(pose_graph.nodes) < 2:
        sys.exit(
            "Error: Registration failed — no frames were successfully registered. "
            "Try lowering --icp_fitness_thresh or increasing --icp_dist_thresh."
        )

    logging.info("Optimizing pose graph...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=args.icp_dist_thresh,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    try:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
    except Exception as e:
        logging.warning(f"Global pose graph optimization failed ({e}); continuing with unoptimized poses.")

    def node_pose(node_idx: int) -> np.ndarray:
        return np.asarray(pose_graph.nodes[node_idx].pose, dtype=np.float64)

    logging.info("Merging point clouds into global map...")
    pcd_combined = o3d.geometry.PointCloud()

    for node_idx, pc_idx in tqdm(list(enumerate(successful_pc_indices)), desc="Merging clouds"):
        _pc_ts, pc_raw = point_clouds[pc_idx]
        T = node_pose(node_idx)

        pcd_world = copy.deepcopy(pc_raw)
        pcd_world.transform(T)
        sensor_origin = T[:3, 3].copy()
        attach_view_rays_as_normals(pcd_world, sensor_origin)

        pcd_combined += pcd_world

    if len(pcd_combined.points) == 0:
        sys.exit("Error: Combined point cloud is empty after merging.")

    if args.level_floor:
        logging.info("Attempting to level the floor...")
        try:
            pcd_tmp = pcd_combined.voxel_down_sample(float(args.voxel_size) * 2.0)
            plane_model, _inliers = pcd_tmp.segment_plane(
                distance_threshold=float(args.voxel_size) * 2.0, ransac_n=3, num_iterations=1000,
            )
            a, b, c, _d = plane_model
            n = np.array([a, b, c], dtype=np.float64)
            n = n / (np.linalg.norm(n) + 1e-12)

            target_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if np.dot(n, target_n) < 0:
                n = -n

            v = np.cross(n, target_n)
            s = np.linalg.norm(v)

            if s > 1e-12:
                cang = float(np.dot(n, target_n))
                vx = np.array(
                    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
                    dtype=np.float64,
                )
                R3 = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - cang) / (s * s))

                pts = np.asarray(pcd_combined.points, dtype=np.float64) @ R3.T
                pcd_combined.points = o3d.utility.Vector3dVector(pts)

                if pcd_combined.has_normals():
                    nr = np.asarray(pcd_combined.normals, dtype=np.float64) @ R3.T
                    pcd_combined.normals = o3d.utility.Vector3dVector(nr)

                logging.info("Floor leveling applied.")
            else:
                logging.info("Map is already level; skipping rotation.")
        except Exception as e:
            logging.warning(f"Floor leveling failed: {e}")

    cleaned = clean_point_cloud(pcd_combined, float(args.voxel_size), do_voxel_downsample=True)

    view_rays = np.asarray(cleaned.normals, dtype=np.float64).copy() if cleaned.has_normals() else None
    estimate_geometric_normals_oriented(cleaned, float(args.voxel_size), view_rays)

    mesh = create_mesh(
        cleaned,
        depth=int(args.poisson_depth),
        min_density_percentile=float(args.min_density_percentile),
        max_vertex_distance=float(args.max_vertex_distance),
        workers=args.workers,
        decimate_target=args.decimate_target,
    )

    # Center mesh XY at origin; shift Z so lowest point sits on the ground plane.
    vertices = np.asarray(mesh.vertices)
    centroid = vertices.mean(axis=0)
    centroid[2] = vertices[:, 2].min()
    mesh.vertices = o3d.utility.Vector3dVector(vertices - centroid)
    logging.info(f"Mesh centered at origin (offset x={centroid[0]:.3f}, y={centroid[1]:.3f}, z={centroid[2]:.3f})")

    # Compute triangle normals required by STL format after all vertex operations are complete.
    mesh.compute_triangle_normals()

    # ----------------------------
    # Gazebo World Export
    # ----------------------------
    logging.info("Exporting to Gazebo format...")
    models_dir = out_dir / "models" / model_name
    meshes_dir = models_dir / "meshes"
    worlds_dir = out_dir / "worlds"

    meshes_dir.mkdir(parents=True, exist_ok=True)
    worlds_dir.mkdir(parents=True, exist_ok=True)

    mesh_stl_path = meshes_dir / "model.stl"
    o3d.io.write_triangle_mesh(str(mesh_stl_path), mesh)
    logging.info(f"Saved mesh: {mesh_stl_path}")

    with open(models_dir / "model.config", "w") as f:
        f.write(CONFIG_TEMPLATE.format(model_name=model_name))

    with open(models_dir / "model.sdf", "w") as f:
        f.write(SDF_TEMPLATE.format(model_name=model_name, gazebo_material=args.gazebo_material))

    with open(worlds_dir / f"{model_name}.world", "w") as f:
        f.write(WORLD_TEMPLATE.format(model_name=model_name))

    logging.info(f"Gazebo environment successfully generated in: {out_dir}")
    logging.info("Done.")


def main():
    p = argparse.ArgumentParser(description="Convert ROS 2 bag directly to a Gazebo simulation environment.")

    p.add_argument("bagpath", help="Path to the ROS 2 bag file.")
    p.add_argument("outputdir", help="Directory to save Gazebo environment outputs.")

    p.add_argument("--model_name", dest="model_name", default="bag_environment",
                   help="Name of the generated Gazebo model.")
    p.add_argument("--gazebo_material", dest="gazebo_material", default="Gazebo/Grey",
                   help="Gazebo material name (e.g. Gazebo/White, Gazebo/Bricks, Gazebo/Wood).")

    p.add_argument("--pc_topic", dest="pc_topic", default="points",
                   help="PointCloud2 topic name.")
    p.add_argument("--odom_topic", dest="odom_topic", default=None,
                   help="Odometry topic (nav_msgs/Odometry).")

    p.add_argument("--voxel_size", dest="voxel_size", type=float, default=0.05,
                   help="Voxel size (meters).")
    p.add_argument("--icp_dist_thresh", dest="icp_dist_thresh", type=float, default=0.2,
                   help="ICP max correspondence distance (meters).")
    p.add_argument("--icp_fitness_thresh", dest="icp_fitness_thresh", type=float, default=0.6,
                   help="Min ICP fitness to accept a frame.")
    p.add_argument("--odom_max_latency", dest="odom_max_latency", type=float, default=0.5,
                   help="Max allowed age of an odometry match in seconds.")

    p.add_argument("--enable_loop_closure", dest="enable_loop_closure", action="store_true", default=False,
                   help="Enable loop closure detection.")
    p.add_argument("--loop_closure_radius", dest="loop_closure_radius", type=float, default=10.0,
                   help="Loop closure search radius (m).")
    p.add_argument("--loop_closure_fitness_thresh", dest="loop_closure_fitness_thresh", type=float, default=0.3,
                   help="Loop closure ICP fitness threshold.")
    p.add_argument("--loop_closure_search_interval", dest="loop_closure_search_interval", type=int, default=10,
                   help="Search for loop closures every N frames.")

    p.add_argument("--poisson_depth", dest="poisson_depth", type=int, default=9,
                   help="Poisson reconstruction depth.")
    p.add_argument("--min_density_percentile", dest="min_density_percentile", type=float, default=1.0,
                   help="Trim bottom density percentile after Poisson reconstruction.")
    p.add_argument("--max_vertex_distance", dest="max_vertex_distance", type=float, default=0.15,
                   help="Trim mesh vertices farther than this from input points (m).")
    p.add_argument("--decimate_target", dest="decimate_target", type=float, default=None,
                   help="Mesh decimation target: <= 1.0 is a ratio, > 1 is an absolute triangle count.")

    p.add_argument("--level_floor", dest="level_floor", action="store_true",
                   help="Attempt to level the floor plane to Z=0.")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel workers for KDTree queries.")

    args = p.parse_args()
    process_bag(args)


if __name__ == "__main__":
    main()

