from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import math

from l5kit.data.zarr_dataset import AGENT_DTYPE

from ..data.labels import PERCEPTION_LABELS
from ..data.filter import filter_agents_by_labels, filter_agents_by_track_id
from ..geometry import rotation33_as_yaw
from .rasterizer import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer


def get_ego_as_agent(frame: np.ndarray) -> np.ndarray:  # TODO this can be useful to have around
    """
    Get a valid agent with information from the frame AV. Ford Fusion extent is used

    Args:
        frame (np.ndarray): the frame we're interested in

    Returns: an agent np.ndarray of the AV

    """
    ego_agent = np.zeros(1, dtype=AGENT_DTYPE)
    ego_agent[0]["centroid"] = frame["ego_translation"][:2]
    ego_agent[0]["yaw"] = rotation33_as_yaw(frame["ego_rotation"])
    ego_agent[0]["extent"] = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
    ego_agent[0]["label_probabilities"] = np.zeros((len(PERCEPTION_LABELS),), dtype=np.float32)
    ego_agent[0]["label_probabilities"][3] = 1.0

    return ego_agent


def draw_boxes(
    raster_size: Tuple[int, int],
    world_to_image_space: np.ndarray,
    agents: np.ndarray,
    color: Union[int, Tuple[int, int, int]],
) -> np.ndarray:
    """
    Draw multiple boxes in one sweep over the image.
    Boxes corners are extracted from agents, and the coordinates are projected in the image plane.
    Finally, cv2 draws the boxes.

    Args:
        raster_size (Tuple[int, int]): Desired output image size
        world_to_image_space (np.ndarray): 3x3 matrix to convert from world to image coordinated
        agents (np.ndarray): array of agents to be drawn
        color (Union[int, Tuple[int, int, int]]): single int or RGB color

    Returns:
        np.ndarray: the image with agents rendered. RGB if color RGB, otherwise GRAY
    """
    if isinstance(color, int):
        im = np.zeros((raster_size[1], raster_size[0]), dtype=np.uint8)
    else:
        im = np.zeros((raster_size[1], raster_size[0], 3), dtype=np.uint8)

    # corners_base_coords = np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    box_world_coords = np.zeros((3, len(agents) * 4))
    corners = np.zeros((3, 4), dtype=np.float32)
    r_m = np.zeros((3, 3), dtype=np.float32)
    r_m[2, 2] = 1.0

    # compute the corner in world-space (start in origin, rotate and then translate)
    for idx, agent in enumerate(agents):
        # corners = corners_base_coords * agent["extent"][:2] / 2  # corners in zero
        # r_m = yaw_as_rotation33(agent["yaw"])
        # box_world_coords[idx] = transform_points(corners, r_m) + agent["centroid"][:2]
        extent = agent["extent"]

        if agent["label_probabilities"][3] == 1:
            # Car - minimum extent..
            # EGO_EXTENT_WIDTH = 1.85
            # EGO_EXTENT_LENGTH = 4.87
            extent[0] = max(3, extent[0])
            extent[1] = max(1.4, extent[1])

        elif agent["label_probabilities"][12] == 1:
            # Cyclists
            extent[0] = max(1.3, extent[0])
            extent[1] = max(0.5, extent[1])

        if agent["label_probabilities"][14] == 1:
            # Increase the pedestrians "size"
            extent = extent * 1.8
        else:
            extent = extent

        extent = extent / 2.0
        centroid = agent["centroid"]

        corners[0, 0] = -extent[0]
        corners[0, 1] = -extent[0]
        corners[0, 2] = extent[0]
        corners[0, 3] = extent[0]
        corners[1, 0] = -extent[1]
        corners[1, 1] = extent[1]
        corners[1, 2] = extent[1]
        corners[1, 3] = -extent[1]

        cy = math.cos(agent["yaw"])
        sy = math.sin(agent["yaw"])

        r_m[0, 0] = cy
        r_m[0, 1] = -sy
        r_m[1, 0] = sy
        r_m[1, 1] = cy

        box_world_coords[:, (4 * idx):(4 * (idx + 1))] = (r_m.dot(corners) + [[centroid[0]], [centroid[1]], [1]])

    # box_image_coords = transform_points(box_world_coords.reshape((-1, 2)), world_to_image_space)
    box_image_coords = world_to_image_space.dot(box_world_coords)[:2, :].T

    # fillPoly wants polys in a sequence with points inside as (x,y)
    box_image_coords = box_image_coords.reshape((-1, 4, 2)).astype(np.int64)
    cv2.fillPoly(im, box_image_coords, color=color)
    return im


class BoxRasterizer(Rasterizer):
    def __init__(
        self,
        raster_size: Tuple[int, int],
        pixel_size: np.ndarray,
        ego_center: np.ndarray,
        filter_agents_threshold: float,
        history_num_frames: int,
    ):
        """

        Arguments:
            raster_size (Tuple[int, int]): Desired output image size
            pixel_size (np.ndarray): Dimensions of one pixel in the real world
            ego_center (np.ndarray): Center of ego in the image, [0.5,0.5] would be in the image center.
            filter_agents_threshold (float): Value between 0 and 1 used to filter uncertain agent detections
            history_num_frames (int): Number of frames to rasterise in the past
        """
        super(BoxRasterizer, self).__init__()
        self.raster_size = raster_size
        self.pixel_size = pixel_size
        self.ego_center = ego_center
        self.filter_agents_threshold = filter_agents_threshold
        self.history_num_frames = history_num_frames

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        world_to_image_space: np.ndarray,
        agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        # this ensures we always end up with fixed size arrays, +1 is because current time is also in the history
        out_im = np.zeros(shape=(self.raster_size[1],
                                 self.raster_size[0],
                                 (self.history_num_frames + 1) * 2),
                          dtype=np.uint8)
        nframe = len(history_frames)

        for i, (frame, agents) in enumerate(zip(history_frames, history_agents)):
            agents = filter_agents_by_labels(agents, self.filter_agents_threshold)
            # note the cast is for legacy support of dataset before April 2020
            av_agent = get_ego_as_agent(frame).astype(agents.dtype)

            if agent is None:
                agents_image = draw_boxes(self.raster_size, world_to_image_space, agents, 255)
                ego_image = draw_boxes(self.raster_size, world_to_image_space, av_agent, 255)
            else:
                agent_ego = filter_agents_by_track_id(agents, agent["track_id"])
                if len(agent_ego) == 0:  # agent not in this history frame
                    agents_image = draw_boxes(self.raster_size, world_to_image_space, np.append(agents, av_agent), 255)
                    ego_image = np.zeros_like(agents_image)
                else:  # add av to agents and remove the agent from agents
                    agents = agents[agents != agent_ego[0]]
                    agents_image = draw_boxes(self.raster_size, world_to_image_space, np.append(agents, av_agent), 255)
                    ego_image = draw_boxes(self.raster_size, world_to_image_space, agent_ego, 255)

            out_im[..., i] = agents_image
            out_im[..., i + nframe] = ego_image

        return out_im.astype(np.float32) / 255

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        """
        get an rgb image where agents further in the past have faded colors

        Args:
            in_im: the output of the rasterize function
            kwargs: this can be used for additional customization (such as colors)

        Returns: an RGB image with agents and ego coloured with fading colors
        """
        hist_frames = in_im.shape[-1] // 2
        in_im = np.transpose(in_im, (2, 0, 1))

        # this is similar to the draw history code
        out_im_agent = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
        agent_chs = in_im[:hist_frames][::-1]  # reverse to start from the furthest one
        agent_color = (0, 0, 1) if "agent_color" not in kwargs else kwargs["agent_color"]
        for ch in agent_chs:
            out_im_agent *= 0.85  # magic fading constant for the past
            out_im_agent[ch > 0] = agent_color

        out_im_ego = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
        ego_chs = in_im[hist_frames:][::-1]
        ego_color = (0, 1, 0) if "ego_color" not in kwargs else kwargs["ego_color"]
        for ch in ego_chs:
            out_im_ego *= 0.85
            out_im_ego[ch > 0] = ego_color

        out_im = (np.clip(out_im_agent + out_im_ego, 0, 1) * 255).astype(np.uint8)
        return out_im
