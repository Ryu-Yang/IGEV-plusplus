import time
import logging
import rerun as rr
import numpy as np
import cv2

log = logging.getLogger()

class Viewer():
    def __init__(self):
        rr.init("Viewer", spawn=True)

        # 创建3D视图
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # 添加坐标系轴
        rr.log("world/axes", rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        ), static=True)

        log.info("🚀 Rerun可视化已启动。\n")

    def _set_timestamp(self, timestamp_ns):
        """
        设置Rerun的时间戳
        
        Args:
            timestamp_ns: 纳秒时间戳
        """
        timestamp_seconds = timestamp_ns / 1e9
        
        rr.set_time("timestamp", timestamp=timestamp_seconds)
        # rr.set_time_seconds("timestamp", seconds = 0)

    def view_image(self, name: str, image: np.ndarray):
        """
        显示图像,
        
        Args:
            name: 图像名字
            images:(HWC)
            timestamps: 
        """
        entity_path = f"image/{name}"
        
        self._set_timestamp(0)

        # retval, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        # if retval:
        #     encoded_image = buffer.tobytes()
        #     rr.log(
        #         f"{entity_path}",
        #         rr.EncodedImage(contents=encoded_image,media_type="image/jpeg")
        #     )
        rr.log(
            f"{entity_path}",
            rr.Image(image)
        )

    def view_depth(self, name: str, depth: np.ndarray):
        """
        显示深度图
        
        Args:
            name: 深度图名称
            depth: 深度图，是 (H, W) 的 numpy 数组
        """
        entity_path = f"world/camera/{name}"

        self._set_timestamp(0)

        max_depth = 5000.0
        # depth_lim = np.clip(depth, 0, max_depth)

        depth_lim = depth.copy()
        depth_lim[depth_lim > max_depth] = 0.0

        # depth_colormap = cv2.applyColorMap(
        #     (depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET
        # )
        # depth_colormap[depth == 0] = [0, 0, 0]  # 无效深度设为黑色

        # rr.log(
        #     f"{entity_path}/depth",
        #     rr.Image(depth_colormap)
        # )

        rr.log(
            f"{entity_path}",
            rr.Pinhole(
                width=depth.shape[1],
                height=depth.shape[0],
                focal_length=500,
            ),
        )

        # Log the tensor.
        rr.log(f"{entity_path}/depth", rr.DepthImage(depth_lim, meter=1_000.0, colormap="viridis"))



    def view_mul_image(self, name: str, images: np.ndarray, timestamps: np.ndarray):
        """
        显示图像,
        
        Args:
            name: 图像名字
            images:(HWC)
            timestamps: 
        """
        log.info(f"view_image: {name}: {len(images)} 个点")

        if len(images) <= 1 or len(timestamps) <= 1:
            return
            
        # 创建轨迹实体路径
        entity_path = f"encoded_images/{name}"
        
        # 按时间顺序记录每个点
        for idx, (image, ts) in enumerate(zip(images, timestamps)):
            self._set_timestamp(ts)

            retval, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            
            if retval:
                encoded_image = buffer.tobytes()
                rr.log(
                    f"{entity_path}",
                    rr.EncodedImage(contents=encoded_image,media_type="image/jpeg")
                )

    def view_mul_depth_maps(self, name: str, depth_maps: np.ndarray, timestamps: np.ndarray):
        """
        显示深度图序列
        
        Args:
            name: 深度图名称
            depth_maps: 深度图列表，每个元素是 (H, W) 的 numpy 数组
            timestamps: 时间戳列表（纳秒）
        """
        log.info(f"view_depth_maps: {name}: {len(depth_maps)} 帧")

        log.info(f"depth_maps.shape: {depth_maps.shape}")
        
        print(f"depth_maps[0]: {depth_maps[0]}")
        if len(depth_maps) <= 1 or len(timestamps) <= 1:
            return

        entity_path = f"depth/{name}"

        for idx, (depth, ts) in enumerate(zip(depth_maps, timestamps)):
            self._set_timestamp(ts)

            # 将深度图转换为可视化图像（归一化到 0-255）
            # 深度范围 0-5m
            max_depth = 10.0
            depth_normalized = np.clip(depth, 0, max_depth) / max_depth
            depth_colormap = cv2.applyColorMap(
                (depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            depth_colormap[depth == 0] = [0, 0, 0]  # 无效深度设为黑色

            rr.log(
                f"{entity_path}",
                rr.Image(depth_colormap)
            )
