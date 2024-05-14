import math
import matplotlib.pyplot as plt

#список вершин vertexes

# print("Введите координаты точки A")
# xA = float(input("x = "))
# yA = float(input("y = "))
#
# print("Введите координаты точки F")
# xF = float(input("x = "))
# yF = float(input("y = "))
#
# print("Введите координаты точки и радиус круга")
# x_circle = float(input("x = "))
# y_circle = float(input("y = "))
# R = float(input("R = "))
xA =-90
yA =90

xF = 150
yF = 250

x_circle=-50
y_circle=100
R=30

class Edge:
    def __init__(self,V1,V2,l):
        self.V1 = V1
        self.V2 = V2
        self.l = l


class Vertex:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.edges=[]
        self.cost = float('inf')
        self.previous = None

    def append_edge(self, edge):
        self.edges.append(edge)

class Circle:
    def __init__(self,x,y,R):
        self.x = x
        self.y = y
        self.R = R
        self.circle_vertexes=[]


    def append_vertex(self, vertex):
        self.circle_vertexes.append(vertex)

def ReserchWay (x,y,v1,v2): #нужна при попадении отрезка на окружность
    return (v1.x<=x<=v2.x or  v2.x<=x<=v1.x) and (v1.y<=y<=v2.y or v2.y<=y<=v1.y)

def Lenght(v1,v2):
    return math.sqrt((v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2)

def LenghtCicles(x1,y1,x2,y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def Tangent(circles, v, vertexes):  # касательные для точки и окружности
    for circle in circles:
        b = math.sqrt((v.x - circle.x) ** 2 + (v.y - circle.y) ** 2)
        th = math.acos(circle.R / b)
        d = math.atan2(v.y - circle.y, v.x - circle.x)
        d1 = d + th
        d2 = d - th
        T1x = circle.x + circle.R * math.cos(d1)
        T1y = circle.y + circle.R * math.sin(d1)
        T2x = circle.x + circle.R * math.cos(d2)
        T2y = circle.y + circle.R * math.sin(d2)

        V1 = Vertex(T1x, T1y)
        V2 = Vertex(T2x, T2y)
        if Point(v,V1,circles, circle):
            vertexes.append(V1)
            circle.append_vertex(V1)

        if Point(v,V2,circles, circle):
            vertexes.append(V2)
            circle.append_vertex(V2)


def TangentCircle(circle1, circle2, vertexes,circles):  # касательные для окружностей
    b = circle1.R + circle2.R
    d = LenghtCicles(circle1.x, circle1.y, circle2.x, circle2.y)
    tetha = math.acos(b / d)  # угол между прямой ,соединяющая центры,и касательной

    alpha = math.atan2(circle2.y - circle1.y,circle2.x - circle1.x)  # угол,на которое происходит смещение относительно горизотальнгой линии центра одного из окружностей.
    # координаты точки касания для 1 окружности
    xT1 = circle1.x + circle1.R * math.cos(tetha + alpha)
    yT1 = circle1.y + circle1.R * math.sin(tetha + alpha)

    xT2 = circle1.x + circle1.R * math.cos(-tetha + alpha)
    yT2 = circle1.y + circle1.R * math.sin(-tetha + alpha)

    alpha = math.atan2(circle1.y - circle2.y, circle1.x - circle2.x)
    x1 = circle2.x + circle2.R * math.cos(tetha + alpha)
    y1 = circle2.y + circle2.R * math.sin(tetha + alpha)

    x2 = circle2.x + circle2.R * math.cos(-tetha + alpha)
    y2 = circle2.y + circle2.R * math.sin(-tetha + alpha)

    V1 = Vertex(xT1, yT1)
    V2 = Vertex(xT2, yT2)
    V3 = Vertex(x1, y1)
    V4 = Vertex(x2, y2)


    if Point(V1, V3, circles, circle1, circle2):
        circle1.append_vertex(V1)
        circle2.append_vertex(V3)
        vertexes.append(V1)
        vertexes.append(V3)

    if Point(V2, V4, circles, circle1,circle2):
        circle1.append_vertex(V2)
        circle2.append_vertex(V4)
        vertexes.append(V2)
        vertexes.append(V4)



    b1 =abs(circle1.R - circle2.R)
    tetha1 = math.acos(b1 / d)
    alpha = math.atan2(circle2.y - circle1.y,circle2.x - circle1.x)  # угол,на которое происходит смещение относительно горизотальнгой линии центра одного из окружностей.
    # координаты точки касания для 1 окружности
    xT1 = circle1.x + circle1.R * math.cos(tetha1 + alpha)
    yT1 = circle1.y + circle1.R * math.sin(tetha1 + alpha)

    xT2 = circle1.x + circle1.R * math.cos(-tetha1 + alpha)
    yT2 = circle1.y + circle1.R * math.sin(-tetha1 + alpha)

    alpha = math.atan2(circle1.y - circle2.y, circle1.x - circle2.x)
    x1 = circle2.x + circle2.R * math.cos(tetha1 + alpha)
    y1 = circle2.y + circle2.R * math.sin(tetha1 + alpha)

    x2 = circle2.x + circle2.R * math.cos(-tetha1 + alpha)
    y2 = circle2.y + circle2.R * math.sin(-tetha1 + alpha)


    V1 = Vertex(xT1, yT1)
    V2 = Vertex(xT2, yT2)
    V3 = Vertex(x1, y1)
    V4 = Vertex(x2, y2)
    if Point(V1, V4, circles, circle1, circle2):
        circle1.append_vertex(V1)
        circle2.append_vertex(V4)
        vertexes.append(V1)
        vertexes.append(V4)

    if Point(V2, V3, circles, circle1, circle2):
        circle1.append_vertex(V2)
        circle2.append_vertex(V3)
        vertexes.append(V2)
        vertexes.append(V3)

def CircleT (circles,vertexes):
    for i in range(0, len(circles)):
        for j in range(i + 1, len(circles)):
            TangentCircle(circles[i], circles[j], vertexes,circles)

def ReserchAlpha (x1,x2,x3,y1,y2,y3):#нахождение угла на окружности по 3м точкам
    a = ((x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1))
    b1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    b2 = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    return math.acos(a/(b1*b2))

def Curve(circles): #кривые на окружности
    for circle in circles:
        for i in range(0,len(circle.circle_vertexes)):
            for j in range(i+1,len(circle.circle_vertexes)):
                v1 = circle.circle_vertexes[i]
                v2= circle.circle_vertexes[j]
                alpha = ReserchAlpha(circle.x,v1.x,v2.x,circle.y,v1.y,v2.y)
                L = (alpha / 180) * math.pi * circle.R
                E= Edge(v1,v2,L)
                v1.append_edge(E)
                v2.append_edge(E)

def PointTangent(v,ax):#отрисовка касательных
    for i in v.edges:
        ax.plot([i.V1.x, i.V2.x], [i.V1.y, i.V2.y], color=[0, 0, 0])


def Point(v1,v2,circles, circle1 = None, circle2 = None):#прямая для 2х точек
    s=True
    for i in circles:
        if i == circle1 or i == circle2:
            continue
        a = i.x
        b = i.y
        if(abs(v1.x - v2.x) < 0.00001):
           k = 10000000000
           c = 10000000000
        else:
            k = (v1.y - v2.y) / (v1.x - v2.x)
            c = (v1.x * v2.y - v2.x * v1.y) / (v1.x - v2.x)
        f = -8 * a * k * c + 8 * a * k * b - 4 * (c ** 2) - 4 * (b ** 2) + 8 * c * b + 4 * (i.R ** 2) - 4 * (k ** 2) * (a ** 2) + 4 * (k ** 2) * (i.R ** 2)
        if f>=0:
            x1 = (2 * (a - k * c + k * b) + math.sqrt(f)) / (2 * (1 + k ** 2))
            x2 = (2 * (a - k * c + k * b) - math.sqrt(f)) / (2 * (1 + k ** 2))
            y1 = k * x1 + c
            y2 = k * x2 + c
            if ReserchWay(x1, y1, v1,v2) or ReserchWay(x2, y2, v1,v2):
                s=False
                break
    if s is True:
        L=Lenght(v1,v2)
        E = Edge(v1,v2,L)
        v1.append_edge(E)
        v2.append_edge(E)
    return(s)

def Grade(a, b):# длина от начала до конца
    (x1, y1) = a
    (x2, y2) = b
    return math.sqrt((x1 - x2) **2 + (y1 - y2)**2)

def Draw(vertexes,ax):#отрисовка
    for v in vertexes:
        PointTangent(v,ax)

def choose_node (reachable,stop):# вспомогательная функция для А* (выбор узла)
    min_cost = float('inf')
    best_node = None

    for node in reachable:
        cost_start_to_node = node.cost
        cost_node_to_goal = Lenght(node,stop)
        total_cost = cost_start_to_node + cost_node_to_goal

        if min_cost > total_cost:
            min_cost = total_cost
            best_node = node

    return best_node

def build_path(to_node):#возврат к изначальному узлу.полный путь
    path = []
    while to_node != None:
        path.append(to_node)
        to_node = to_node.previous
    return path

def Find(node,explored):#ищем верщины ,которые не исследованные ,из данной сейчас вершины
    found=[]
    for edge in node.edges:
        v=None
        if node != edge.V1:
            v = edge.V1
        else:
            v = edge.V2
        if v not in explored:
            found.append(v)
    return found

def A_star(start, stop):#алгоритм А*
    start.cost = 0
    reachable = [start]
    explored = []

    while len(reachable) > 0:
        node = choose_node(reachable,stop)

        if node == stop:
            return build_path(stop)

        reachable.remove(node)
        explored.append(node)

        new_reachable = Find(node,explored)

        for adjacent in new_reachable:
            if adjacent not in reachable:
                reachable.append(adjacent)

            if node.cost + Lenght(node,adjacent) < adjacent.cost:
                adjacent.previous = node
                adjacent.cost = node.cost + Lenght(node,adjacent)

    return None

def DrawWay(result,ax): #отрисовка А*
    for i in range(len(result)-1):
        ax.plot([result[i].x, result[i+1].x], [result[i].y, result[i+1].y], color=[1, 0, 0])
        ax.plot(result[i].x, result[i].y, 'o', color=[1, 0, 0])
    ax.plot(result[-1].x, result[-1].y, 'o', color=[1, 0, 0])

A = Vertex(xA,yA)
F = Vertex(xF,yF)
Circle1 = Circle(x_circle,y_circle,R)
Circle2 = Circle(100,200,50)
Circle3 = Circle(0,180,30)
circles=[Circle1,Circle2,Circle3]

vertexes = [A, F]
Tangent(circles,A,vertexes)
Tangent(circles,F,vertexes)
CircleT(circles,vertexes)
Curve(circles)
Point(A,F,circles)
Result = A_star(A,F)



fig, ax = plt.subplots(figsize=(10,10))
ax = plt.gca()
ax.set_aspect('equal')
circle1 = plt.Circle((Circle1.x, Circle1.y), Circle1.R, fill=False)
circle2 = plt.Circle((Circle2.x, Circle2.y), Circle2.R, fill=False)
circle3 = plt.Circle((Circle3.x, Circle3.y), Circle3.R, fill=False)
a = ax.plot(A.x, A.y, 'o', color=[0, 0, 0])[0]
f = ax.plot(F.x, F.y, 'o', color=[0, 0, 0])[0]
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
#Draw(vertexes,ax)
DrawWay(Result,ax)
plt.show()
