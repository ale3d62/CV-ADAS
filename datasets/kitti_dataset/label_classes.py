from dataclasses import dataclass, field
from typing import List
import xml.etree.ElementTree as ET


@dataclass
class Pose:
    tx: float
    ty: float
    tz: float
    rx: float
    ry: float
    rz: float

    state: int
    occlusion: int
    truncation: int


@dataclass
class Tracklet:
    object_type: str
    h: float
    w: float
    l: float
    first_frame: int

    poses: List[Pose] = field(default_factory=list)



def parse_kitti_tracklets(xml_path: str) -> List[Tracklet]:

    tree = ET.parse(xml_path)
    root = tree.getroot()

    tracklets = []

    for item in root.findall(".//tracklets/item"):

        object_type = item.find("objectType").text
        h = float(item.find("h").text)
        w = float(item.find("w").text)
        l = float(item.find("l").text)
        first_frame = int(item.find("first_frame").text)

        tracklet = Tracklet(
            object_type=object_type,
            h=h, w=w, l=l,
            first_frame=first_frame
        )

        poses_node = item.find("poses")

        for pose_item in poses_node.findall("item"):

            pose = Pose(
                tx=float(pose_item.find("tx").text),
                ty=float(pose_item.find("ty").text),
                tz=float(pose_item.find("tz").text),

                rx=float(pose_item.find("rx").text),
                ry=float(pose_item.find("ry").text),
                rz=float(pose_item.find("rz").text),

                state=int(pose_item.find("state").text),
                occlusion=int(pose_item.find("occlusion").text),
                truncation=int(pose_item.find("truncation").text),
            )

            tracklet.poses.append(pose)

        tracklets.append(tracklet)

    return tracklets
