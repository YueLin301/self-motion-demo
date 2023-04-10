from spatialmath import SE3
from roboticstoolbox import RevoluteDH, DHRobot
import numpy as np
import math

pi = math.pi


def self_motion_swing():
    lenoflink = 24
    DHs = []
    for i in range(int(lenoflink / 2)):
        DHs.append(RevoluteDH(a=1, alpha=pi / 2))
        DHs.append(RevoluteDH(a=1, alpha=-pi / 2))
    DHs.append(RevoluteDH(a=0))
    lenoflink += 1

    snake = DHRobot(DHs, name="RLsnake")

    # ====================

    T0 = SE3(0, 18, 10)
    sol = snake.ikine_LM(T0)
    snake.q = sol.q

    I = np.eye(lenoflink)
    qs = [snake.q]
    for i in range(100):
        v_raw = T0.t - snake.fkine(qs[-1]).t
        # v_raw = np.array([0, 0, 0])
        v = np.concatenate((v_raw, np.array([0, 0, 0])))

        j = snake.jacobe(qs[-1])
        j_pinv = np.linalg.pinv(j)

        # k = -0.05
        k = -0.2
        dH = np.random.random(lenoflink) * k
        dH = dH.T
        dq = j_pinv @ v + (I - j_pinv @ j) @ dH
        print(j @ (I - j_pinv @ j))

        q = qs[-1] + dq
        qs.append(q)
        print(snake.fkine(q))

    qs = np.array(qs)
    env = snake.plot(qs, dt=0.01, movie='./gifs/self_motion_swing.gif')
    # env.hold()


def self_motion_regularizer():
    lenoflink = 6
    DHs = []
    for i in range(int(lenoflink / 2)):
        DHs.append(RevoluteDH(a=1, alpha=pi / 2))
        DHs.append(RevoluteDH(a=1, alpha=-pi / 2))
    DHs.append(RevoluteDH(a=0))
    lenoflink += 1

    snake = DHRobot(DHs, name="RLsnake")

    # ====================

    T0 = SE3(2 * math.sqrt(3), 3, 0)
    T1 = SE3(2 * math.sqrt(3), -3, 0)

    sol = snake.ikine_LM(T0)
    snake.q = sol.q

    v = np.array([[0], [0], [0], [0], [0], [0]])
    I = np.eye(7)
    T_all = snake.fkine_all(snake.q)
    qs = [snake.q]
    # print(T1.t)
    # print(T_all[-1].t)
    # while cal_distance(T_all[-1].t, T1.t) > 1:
    for i in range(100):
        j = snake.jacobe(qs[-1])
        j_pinv = np.linalg.pinv(j)

        # # 正常走
        # dq = j_pinv @ v

        # 限制中间的那个关节伸直
        k = -0.1
        dH = np.array([[0], [0], [0], [qs[-1][3]], [qs[-1][4]], [0], [0]]) * k
        dq = j_pinv @ v + (I - j_pinv @ j) @ dH

        dq = dq[:, 0]
        # print(dq)

        q = qs[-1] + dq
        qs.append(q)
        T_all = snake.fkine_all(q)
        print(snake.fkine(q))

    qs = np.array(qs)
    env = snake.plot(qs, dt=0.01, movie='./gifs/self_motion_regularizer.gif')
    # env.hold()


if __name__ == '__main__':
    self_motion_swing()
    self_motion_regularizer()

    print('all done.')
