import numpy as np
import cv2
from enum import Enum
from typing import List

cos_angle = lambda x: np.math.cos(x / 180 * np.pi)
sin_angle = lambda x: np.math.sin(x / 180 * np.pi)
tan_angle = lambda x: np.math.tan(x / 180 * np.pi)


def rotation(angle, n):
    '''
    罗德里格斯旋转

    angle: 为旋转角度
    n:     旋转轴[3x1]
    假定旋转是过原点的，起点是原点，n是旋转轴
    '''

    n = np.array(n).reshape(3, 1)
    nx, ny, nz = n[0, 0], n[1, 0], n[2, 0]
    M = np.array([
        [0, -nz, ny],
        [nz, 0, -nx],
        [-ny, nx, 0]
    ])
    R = np.eye(4)
    R[:3, :3] = cos_angle(angle) * np.eye(3) + (1 - cos_angle(angle)) * n @ n.T + sin_angle(angle) * M
    return R


def translate(tx=0, ty=0, tz=0):
    '''
    平移矩阵

    tx：  x轴的平移量
    ty：  y轴的平移量
    tz：  z轴的平移量
    '''
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])


def get_view_matrix(e, g, t):
    '''
    获取视图矩阵

    e:  相机的原点
    g:  相机的朝向
    t:  相机头上方向的朝向
    '''
    e = np.array(e).reshape(3, 1)
    g = np.array(g).reshape(3, 1)
    t = np.array(t).reshape(3, 1)
    
    T_view = np.array([
        [1, 0, 0, -e[0, 0]],
        [0, 1, 0, -e[1, 0]],
        [0, 0, 1, -e[2, 0]],
        [0, 0, 0, 1],
    ])
    
    g_cross_t = np.cross(g.T, t.T).T
    R_view = np.array([
        [g_cross_t[0, 0], t[0, 0], -g[0, 0], 0],
        [g_cross_t[1, 0], t[1, 0], -g[1, 0], 0],
        [g_cross_t[2, 0], t[2, 0], -g[2, 0], 0],
        [0, 0, 0, 1]
    ]).T
    return R_view @ T_view


def get_perspective_matrix(eye_fov, aspect_ratio, near, far):
    '''
    透视投影变换

    eye_fov：      视场角
    aspect_ratio： 视窗的宽高比
    near：         近平面
    far：          远平面
    '''
    
    t = -abs(near) * tan_angle(eye_fov / 2.0)
    r = t * aspect_ratio
    
    return np.array([
        [near / r, 0, 0, 0],
        [0, near / r, 0, 0],
        [0, 0, (near + far) / (near - far), -2 * near * far / (near - far)],
        [0, 0, 1, 0]
    ])


def get_viewport_matrix(width, height, near, far):
    '''
    视口矩阵

    width：  视口宽度
    height： 视口高度
    near：   近平面
    far：    远平面
    '''
    return np.array([
        [width / 2, 0, 0, width / 2],
        [0, -height / 2, 0, height / 2],
        [0, 0, -(near - far) / 2.0, (near + far) / 2.0],
        [0, 0, 0, 1],
    ])


def get_grid_box(w=3, h=3, y=0, n=10, basex=None, basey=None):
    '''
    获取mesh框坐标，返回的是对应的网格坐标
    
    w：    网格的宽度
    h:     网格的高度
    y：    网格坐标的y轴
    n：    网格的密度
    basex：网格x的起点，如果为None，则是w / 2
    basey：网格y的起点，如果为None，则是h / 2
    '''

    basex = w / 2 if basex is None else basex
    basey = h / 2 if basey is None else basey
    x, z = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, n))
    return np.stack([x - basex, np.full_like(x, y), z - basey], axis=-1)


def get_axes_lines():
    '''
    获取坐标系的直线坐标
    '''
    return np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])[[0, 1, 0, 2, 0, 3]]


def get_relative_position(mvp, w, h):
    '''
    获取相对位置，平面坐标，返回[-1, +1]

    mvp：  物体对应的mvp矩阵
    w：    空间的宽度
    h：    空间的高度
    '''
    t = mvp[:, 3]
    t = t / t[3]
    return t[0] / (w / 2), -t[2] / (h / 2)


def get_agent(s=0.5):
    '''
    获取一个agent，以box形式表示，返回其每一个线条的坐标
    坐标点分为2层，第一层是底层，顺序为0 1 2 3，分别是
    0    1

    3    2

    第二层为顶层，顺序是4 5 6 7，分别是
    4    5

    6    7
    '''
    return np.array([
        -s/2, 0, -s/2,
        +s/2, 0, -s/2,
        +s/2, 0, +s/2,
        -s/2, 0, +s/2,
        -s/2, s, -s/2,
        +s/2, s, -s/2,
        +s/2, s, +s/2,
        -s/2, s, +s/2,
    ]).reshape(-1, 3)[[0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7, 3, 6, 2, 7, 0, 2, 1, 3]]


def draw_meshgrid(image, mvp, grid, near, color=(0, 255, 0), width=1, limit=10000):
    '''
    绘制网格

    image：   需要绘制的图像
    mvp：     对应的mvp矩阵
    grid：    网格坐标信息[n, m, 3]
    near：    近平面
    color：   绘制的颜色
    width：   绘制的线宽
    limit：   坐标的限制，超出范围的拒绝绘制
    '''

    h, w   = image.shape[:2]
    ph, pw = grid.shape[:2]
    n      = ph * pw
    grid   = np.concatenate([grid.reshape(n, 3), np.ones((n, 1))], axis=-1)
    proj   = mvp @ grid.T
    points = (proj / proj[3]).T.reshape(ph, pw, 4)

    for iw in range(pw):
        for ih in range(ph-1):
            x1, y1, z1, w1 = points[ih, iw]
            x2, y2, z2, w2 = points[ih+1, iw]
            if z1 >= near or z2 >= near or abs(x1) > limit or abs(y1) > limit or abs(x2) > limit or abs(y2) > limit:
                continue

            p0 = tuple(map(int, [x1*16, y1*16]))
            p1 = tuple(map(int, [x2*16, y2*16]))
            cv2.line(image, p0, p1, color, width, 16, 4)

    for ih in range(ph):
        for iw in range(pw-1):
            x1, y1, z1, w1 = points[ih, iw]
            x2, y2, z2, w2 = points[ih, iw+1]
            if z1 >= near or z2 >= near or abs(x1) > limit or abs(y1) > limit or abs(x2) > limit or abs(y2) > limit:
                continue

            p0 = tuple(map(int, [x1*16, y1*16]))
            p1 = tuple(map(int, [x2*16, y2*16]))
            cv2.line(image, p0, p1, color, width, 16, 4)


def draw_lines(image, mvp, points, near, color=(0, 255, 0), width=1, limit=10000):
    '''
    绘制直线

    image：   需要绘制的图像
    mvp：     对应的mvp矩阵
    points：  直线对应的点坐标[(nx2), 3]对应于n条直线
    near：    近平面
    color：   绘制的颜色
    width：   绘制的线宽
    limit：   坐标的限制，超出范围的拒绝绘制
    '''
    h, w   = image.shape[:2]
    n      = len(points)
    points = np.concatenate([points, np.ones((n, 1))], axis=-1)
    proj   = mvp @ points.T
    points = (proj / proj[3]).T

    for p0, p1 in zip(points[::2], points[1::2]):

        x1, y1, z1, w1 = p0
        x2, y2, z2, w2 = p1
        if z1 >= near or z2 >= near or abs(x1) > limit or abs(y1) > limit or abs(x2) > limit or abs(y2) > limit:
            continue

        p0 = tuple(map(int, [x1 * 16, y1 * 16]))
        p1 = tuple(map(int, [x2 * 16, y2 * 16]))
        cv2.line(image, p0, p1, color, width, 16, 4)


def draw_arrowedline(image, mvp, points, near, color=(0, 255, 0), width=1, limit=10000, tipsize=0.3):
    '''
    绘制箭头

    image：   需要绘制的图像
    mvp：     对应的mvp矩阵
    points：  直线对应的点坐标[(nx2), 3]对应于n条直线
    near：    近平面
    color：   绘制的颜色
    width：   绘制的线宽
    limit：   坐标的限制，超出范围的拒绝绘制
    tipsize： 箭头的头对应大小
    '''
    h, w   = image.shape[:2]
    n      = len(points)
    points = np.concatenate([points, np.ones((n, 1))], axis=-1)
    proj   = mvp @ points.T
    points = (proj / proj[3]).T

    for p0, p1 in zip(points[::2], points[1::2]):

        x1, y1, z1, w1 = p0
        x2, y2, z2, w2 = p1
        if z1 >= near or z2 >= near or abs(x1) > limit or abs(y1) > limit or abs(x2) > limit or abs(y2) > limit:
            continue

        p0 = tuple(map(int, [x1 * 16, y1 * 16]))
        p1 = tuple(map(int, [x2 * 16, y2 * 16]))
        cv2.arrowedLine(image, p0, p1, color, width, 16, 4, tipsize)


def fill_meshgrid(image, mvp, grid, near, color=(0, 255, 0), width=-1, limit=10000):
    '''
    填充网格

    image：   需要绘制的图像
    mvp：     对应的mvp矩阵
    grid：    网格坐标信息[n, m, 3]
    near：    近平面
    color：   绘制的颜色
    width：   绘制的线宽
    limit：   坐标的限制，超出范围的拒绝绘制
    '''

    h, w   = image.shape[:2]
    ph, pw = grid.shape[:2]
    n      = ph * pw
    grid   = np.concatenate([grid.reshape(n, 3), np.ones((n, 1))], axis=-1)
    proj   = mvp @ grid.T
    points = (proj / proj[3]).T.reshape(ph, pw, 4)

    for iw in range(pw-1):
        for ih in range(ph-1):
            x1, y1, z1, w1 = points[ih  , iw  ]
            x2, y2, z2, w2 = points[ih  , iw+1]
            x3, y3, z3, w3 = points[ih+1, iw+1]
            x4, y4, z4, w4 = points[ih+1, iw  ]
            if z1 >= near or z2 >= near or abs(x1) > limit or abs(y1) > limit or abs(x2) > limit or abs(y2) > limit:
                continue

            if z3 >= near or z4 >= near or abs(x3) > limit or abs(y3) > limit or abs(x4) > limit or abs(y4) > limit:
                continue

            ps = np.array([x1, y1, x2, y2, x3, y3, x4, y4], dtype=np.int32).reshape(1, -1, 2)
            cv2.fillPoly(image, ps, color)


def ease_outback(x: float):
    c1 = 1.70158;
    c3 = c1 + 1;
    return 1 + c3 * pow(x - 1, 3) + c1 * pow(x - 1, 2);



class ActionType(Enum):
    Nothing     = 0
    Forward     = 1
    Backward    = 2
    TurnLeft    = 3
    TurnRight   = 4


class ThreeDAgent:

    def __init__(self):

        self.px           = 0
        self.py           = 0
        self.pz           = 5
        self.fov          = 45
        self.near         = 0.1
        self.far          = 50
        self.scene_width  = 600
        self.scene_height = 600
        self.top_width    = 100
        self.top_height   = 100
        self.time         = 0
        self.playground_w = 13
        self.playground_h = 15
        self.angle        = 0
        self.line_velocity    = 0.01
        self.current_line_vel = 0
        self.angular_velocity = 0.05
        self.current_angular_vel = 0
        self.location     = self.px, self.py
        self.prime_camera = get_view_matrix([0, +1, +2], [0,  0, -1], [0, 1, 0])
        self.mask_camera  = get_view_matrix([0, +0.8, -0.5], [0,  0, -1], [0, 1, 0])
        self.top_camera   = get_view_matrix([0, 20, 0],  [0, -1, 0], [0, 0, -1])
        self.prime_image  = np.zeros((self.scene_height, self.scene_width, 3), dtype=np.uint8)
        self.top_image    = np.zeros((self.top_height, self.top_width, 3), dtype=np.uint8)
        self.front_camera_mask = np.zeros((self.scene_height, self.scene_width, 1), dtype=np.uint8)

        self.projection       = get_perspective_matrix(self.fov, 1, self.near, self.far)
        self.prime_view_port  = get_viewport_matrix(self.scene_width, self.scene_height, self.near, self.far)
        self.top_view_port    = get_viewport_matrix(self.top_width, self.top_height, self.near, self.far)
        self.prime_mvp        = self.prime_view_port @ self.projection @ rotation(15, [1, 0, 0])
        self.top_mvp          = self.top_view_port   @ self.projection @ self.top_camera
        self.agent            = translate(self.px, self.py, self.pz)
        self.agent_appearance = get_agent(0.5)
        self.agent_direction  = np.array([[0, 0.25, 0], [0, 0.25, -0.5]])
        self.playground_grid  = get_grid_box(self.playground_w, self.playground_h, 0, 20)


    def reset(self):

        self.agent          = translate(self.px, self.py, self.pz)
        self.prime_image[:] = 0
        self.top_image[:]   = 0
        self.front_camera_mask[:] = 0
        self.time           = 0
        self.current_line_vel    = 0
        self.current_angular_vel = 0
        self.location       = self.px, self.py


    def draw_sysinfo(self, image):

        ax, ay = self.location
        cv2.putText(image, f"Time {self.time}", (15, 30), 5, 1, (0, 255, 0), 1, 16)
        cv2.putText(image, f"Agent x:{ax:.3f}, y:{ay:.3f}, a:{self.angle:.3f}", (15, 60), 5, 1, (0, 255, 0), 1, 16)
        cv2.putText(image, f"Velocity L:{self.current_line_vel:.3f}, A:{self.current_angular_vel:.3f}", (15, 90), 5, 1, (0, 255, 0), 1, 16)


    def render_topview(self):

        self.top_image  [:] = 0
        fill_meshgrid(self.top_image, self.top_mvp, self.playground_grid, self.near, (100, 200, 100))
        draw_meshgrid(self.top_image, self.top_mvp, self.playground_grid, self.near, (255, 255, 255))
        draw_lines(self.top_image,    self.top_mvp @ self.agent, self.agent_appearance, self.near, (255, 0, 255), 2)
        draw_arrowedline(self.top_image,    self.top_mvp @ self.agent, self.agent_direction, self.near, (0, 255, 255), 2)
        return self.top_image


    def render(self):

        self.render_topview()
        self.prime_image[:] = 0

        inv_agent = np.linalg.inv(self.agent)
        mvp       = self.prime_mvp @ self.prime_camera @ inv_agent
        fill_meshgrid(self.prime_image, mvp, self.playground_grid, self.near, (100, 200, 100))
        draw_meshgrid(self.prime_image, mvp, self.playground_grid, self.near, (255, 255, 255))
        draw_lines(self.prime_image,    mvp @ self.agent, self.agent_appearance, self.near, (255, 0, 255), 2)

        top_margin = 20
        self.prime_image[top_margin:top_margin+self.top_height, -self.top_width-top_margin:-top_margin] = self.top_image
        cv2.rectangle(self.prime_image, (self.scene_width - top_margin - self.top_width, top_margin), (self.scene_width - top_margin, top_margin + self.top_height), (255, 255, 255), 2)
        self.draw_sysinfo(self.prime_image)
        return self.prime_image


    def render_front_camera_mask(self):

        self.front_camera_mask[:] = 0
        inv_agent = np.linalg.inv(self.agent)
        mvp       = self.prime_mvp @ self.mask_camera @ inv_agent
        fill_meshgrid(self.front_camera_mask, mvp, self.playground_grid, self.near, (255, 255, 255))
        return self.front_camera_mask


    def step(self, actions : List[ActionType]):

        self.time += 1
        if isinstance(actions, ActionType):
            actions = [actions]

        action_matrix = np.eye(4)

        has_move, has_rotate = False, False
        for action in actions:

            has_move   = has_move   or action == ActionType.Forward or action == ActionType.Backward
            has_rotate = has_rotate or action == ActionType.TurnLeft or action == ActionType.TurnRight

            if action == ActionType.Forward:
                self.current_line_vel = max(self.current_line_vel - self.line_velocity, -0.05)
            elif action == ActionType.Backward:
                self.current_line_vel = min(self.current_line_vel + self.line_velocity, 0.05)
            elif action == ActionType.TurnLeft:
                self.current_angular_vel = max(self.current_angular_vel - self.angular_velocity, -1)
            elif action == ActionType.TurnRight:
                self.current_angular_vel = min(self.current_angular_vel + self.angular_velocity, 1)

        if not has_move:
            self.current_line_vel -= self.current_line_vel * 0.05

        if not has_rotate:
            self.current_angular_vel -= self.current_angular_vel * 0.2

        if abs(self.current_line_vel) > 1e-5:
            action_matrix = action_matrix @ translate(tz=self.current_line_vel)

        if abs(self.current_angular_vel) > 1e-5:
            action_matrix = action_matrix @ rotation(self.current_angular_vel, [0, 1, 0])

        self.agent    = self.agent @ action_matrix
        self.location = get_relative_position(self.agent, self.playground_w, self.playground_h)
        self.angle    = np.math.acos(max(-1, min(1, self.agent[0, 0]))) / np.pi * 180
        return self.location, self.angle, self.current_line_vel, self.current_angular_vel


td = ThreeDAgent()

# pip install pygame
import pygame

pygame.init()
screen_large = np.zeros((600, 600 * 2, 3), np.uint8)
screen       = pygame.display.set_mode((screen_large.shape[1], screen_large.shape[0]))
pygame.display.set_caption("Robot")

while True:
    prime = td.render()
    mask  = td.render_front_camera_mask()

    screen_large[:, :600]            = prime
    screen_large[:, 600:, [0, 1, 2]] = mask
    surf = pygame.surfarray.make_surface(screen_large.transpose(1, 0, 2))
    screen.blit(surf, (0, 0))
    pygame.display.update()

    key = pygame.key.get_pressed()
    pygame.event.pump()

    actions = []
    if key[pygame.K_w]:
        actions.append(ActionType.Forward)
    elif key[pygame.K_s]:
        actions.append(ActionType.Backward)

    if key[pygame.K_e]:
        actions.append(ActionType.TurnLeft)
    elif key[pygame.K_q]:
        actions.append(ActionType.TurnRight)

    if key[pygame.K_c]:
        td.reset()
        continue

    if key[pygame.K_z]:
        break

    td.step(actions)