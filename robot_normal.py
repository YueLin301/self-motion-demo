from spatialmath import *
from roboticstoolbox import *
from roboticstoolbox.backends.PyPlot import PyPlot
from spatialgeometry import *
import numpy as np
import math

pi = math.pi


def cal_distance(a, b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i] - b[i]) ** 2
    return sum ** 0.5


def my_jacobe_ikine():
    lenoflink = 6
    DHs = []
    for i in range(int(lenoflink / 2)):
        DHs.append(RevoluteDH(a=1, alpha=pi / 2))
        DHs.append(RevoluteDH(a=1, alpha=-pi / 2))
    DHs.append(RevoluteDH(a=0))
    lenoflink += 1

    snake = DHRobot(DHs, name="RLsnake")

    T0 = SE3(2 * math.sqrt(3), 3, 0)
    T1 = SE3(2 * math.sqrt(3), -3, 0)

    sol = snake.ikine_LM(T0)
    snake.q = sol.q
    # env = snake.plot(snake.q)
    # env.hold()

    v = np.array([[0], [-0.03], [0], [0], [0], [0]])
    # v = np.array([[0], [0], [0], [0], [0], [0]])
    I = np.eye(7)
    T_all = snake.fkine_all(snake.q)
    qs = [snake.q]

    while cal_distance(T_all[-1].t, T1.t) > 1:
        j = snake.jacobe(qs[-1])
        j_pinv = np.linalg.pinv(j)

        dq = j_pinv @ v
        dq = dq[:, 0]

        q = qs[-1] + dq
        qs.append(q)
        T_all = snake.fkine_all(q)
        print(snake.fkine(q))

    qs = np.array(qs)
    env = snake.plot(qs, dt=0.01, movie='./gifs/my_jacobe_ikine.gif')
    # env.hold()


def ikine_toolbox():
    lenoflink = 6
    DHs = []
    for i in range(int(lenoflink / 2)):
        DHs.append(RevoluteDH(a=1, alpha=pi / 2))
        DHs.append(RevoluteDH(a=1, alpha=-pi / 2))
    DHs.append(RevoluteDH(a=0))
    lenoflink += 1

    snake = DHRobot(DHs, name="RLsnake")

    t = np.arange(0, 2, 0.010)
    T0 = SE3(3 * math.sqrt(3), 3, 0)
    T1 = SE3(3 * math.sqrt(3), -3, 0)
    Ts = ctraj(T0, T1, t)
    print(Ts)

    sol = snake.ikine_LM(Ts)

    env = PyPlot()
    env.launch()

    env = snake.plot(sol.q, dt=0.01, fig=env.fig, movie='./gifs/ikine_toolbox.gif')
    # env.hold()


if __name__ == '__main__':
    # my_jacobe_ikine()
    ikine_toolbox()

    print('all done.')
