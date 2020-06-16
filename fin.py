import json
import matplotlib.pyplot as plt
import numpy as np
from brokenaxes import brokenaxes

class Film:
    """
    电影类, 存储id, 标题, 年份, 类型, 星级等信息
    """

    def __init__(self, item):
        self.id = item['_id']['$oid']
        self.title = item['title']
        self.year = item['year']
        self.type = item['type'].split(',')
        self.star = float(item['star'])


class Vertex:
    """
    节点类, 以演员为节点. 类成员包括:
    - 演员姓名(str)
    - 共同出演者(共同出演者(key, Vertex)和共同出演的电影(value, List[Film])构成的字典
    - 演员出演的电影字典(key: Film.id, Value: Film)
    - 用于广度优先搜索的参量
    - 用于建立队列的参量
    """

    def __init__(self, actor):
        self.actor = actor
        self.neighbour = {}
        self.films = {}
        # 用于广度优先搜索
        self.distance = 0
        self.predecessor = None
        self.colour = 0  # 0未发现, 1已发现, 2完成探索
        # 用于bfs时建立队列
        self.Prev = None
        self.Next = None

    def addNeighbour(self, vert, film):
        """ 和vert通过film建立联系. """
        if vert in self.neighbour:
            self.neighbour[vert].append(film)
        else:
            self.neighbour[vert] = [film]

    def getConnections(self):
        """ 获取所有有关联的节点 """
        return self.neighbour.keys()

    def addFilm(self, film):
        """ 为节点添加出演电影 """
        self.films[film.id] = film


class Graph:
    """ 
    图类. 含有如下成员:
    - 包含所有节点的字典(以演员名字为key, 以演员节点为value)
    - 节点数
    - 连通分支, list(dict(演员姓名: 演员节点)), 每一个字典对应一个连通分支, 按大小排序
    """

    def __init__(self, data):
        """ 从数据data中构建图并计算连通分支 """
        self.vertlist = {}
        self.numVertices = 0
        self.cnct_comp = []
        for item in data:
            self.addFromItem(item)
        self.set_cnct_comp()

    def addVertex(self, actor):
        """ 向图中添加姓名为actor的演员, 并返回演员节点 """
        if actor in self.vertlist:  # 已有就不再添加, 直接返回
            return self.vertlist[actor]
        else:
            self.numVertices += 1
            newVertex = Vertex(actor)
            self.vertlist[actor] = newVertex
            return newVertex

    def addEdge(self, actor1, actor2, film):
        """ 在姓名为actor1, actor2的演员之间通过film建立联系 """
        v1 = self.vertlist[actor1]
        v2 = self.vertlist[actor2]
        v1.addNeighbour(v2, film)
        v2.addNeighbour(v1, film)

    def __contain__(self, actor):
        return actor in self.vertlist

    def addFromItem(self, item):
        """ 从一个电影条目item中添加节点和边 """
        actors = item['actor'].split(',')  # 获取这个电影的演员列表
        n = len(actors)
        film = Film(item)  # 从条目构建电影
        for i in range(n):  # 遍历参与这个电影的所有演员
            if actors[i]!='':
                v = self.addVertex(actors[i])  # 添加演员节点
                v.addFilm(film)  # 为节点添加电影
                for j in range(i):  # 遍历参与这个电影的, 当前这个演员之前的演员
                    if actors[j]!='':
                        self.addEdge(actors[j], actors[i], film)  # 在两个演员之间添加边

    def bfs(self, start, d=None, end=None):
        """ 
        广度优先搜索
        - start: 起始节点
        - d: 存储搜索过的节点的字典, 缺省值为None. 这一参数用于构建连通分支
        - end: 结束搜索的节点, 缺省值为None, 若缺省, 则搜遍start所在连通分支
        - 返回最大搜索深度 
        """
        start.distance = 0  # 起点的搜索距离为0
        start.predecessor = None  # 起点的前驱为None
        vertQueue = Queue()  # 建立队列
        vertQueue.enqueue(start)  # 起点入队
        distance = 0  # 搜索深度, 初始值为0
        while (vertQueue.length > 0):  # 循环, 当队列为空时终止
            currentVert = vertQueue.dequeue()  # 队首出队, 作为当前节点
            distance = currentVert.distance  # 搜索深度为当前节点的距离
            if d is not None:  # d不为None, 则将当前节点加入d
                d[currentVert.actor] = currentVert
            for nbr in currentVert.getConnections():  # 遍历所有相邻的节点
                if nbr.colour == 0:  # 如果初次遇到, 标记为正在搜索, 距离为当前节点加一, 标记前驱, 入队
                    nbr.colour = 1
                    nbr.distance = currentVert.distance+1
                    if nbr == end:  # 如果是终止节点, 停止搜索, 返回搜索深度(距离)
                        return nbr.distance
                    nbr.predecessor = currentVert
                    vertQueue.enqueue(nbr)
            currentVert.colour = 2  # 遍历完所有相邻节点, 标记为已搜索完成
        return distance

    def set_cnct_comp(self):
        """ 为连通分支这一成员变量正确赋值 """
        for key in self.vertlist:  # 遍历图中的节点
            vert = self.vertlist[key]
            if vert.colour == 0:  # 如果为搜索过, 以该节点为起点bfs, 并将所有搜索的节点存入d, 将d存入self.cnct_comp
                d = {}
                self.bfs(vert, d)
                self.cnct_comp.append(d)
        self.cnct_comp.sort(key=lambda list1: len(list1),
                             reverse=True)  # 按连通分支规模进行排序

    def bfs_reset(self, index=None):
        """
        bfs之后的重置, 将所有节点置于未搜索状态, 距离置零
        - 重置的范围是第index个连通分支. index缺省值为None, 如是, 则重置图中全部节点
        """
        if index is None:
            for k in self.vertlist:
                v = self.vertlist[k]
                v.colour = 0
                v.distance = 0
        else:
            for k in self.cnct_comp[index]:
                v = self.vertlist[k]
                v.colour = 0
                v.distance = 0

    def diameter(self, index):
        """ 返回第index个连通分支的直径 """
        d = self.cnct_comp[index]
        R = 0
        for k in d:
            self.bfs_reset(index)
            r = self.bfs(self.vertlist[k])
            if r > R:
                R = r
        return R

    def films_in_cont(self, index):
        """ 以字典的形式(Film.id: Film)返回第index个连通分支内的所有电影 """
        d = self.cnct_comp[index]
        films = {}
        for k in d:
            for film in self.vertlist[k].films.values():
                films[film.id] = film
        return films


class Queue:
    """ 队列类, 用于bfs """

    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0

    def enqueue(self, vert):
        """ 入队 """
        if self.length == 0:
            self.head = self.tail = vert
        else:
            vert.Prev = self.tail
            self.tail.Next = vert
            self.tail = vert
        self.length += 1

    def dequeue(self):
        """ 出队 """
        self.length -= 1
        if self.length == 0:
            vert = self.head
            self.head = self.tail = None
            return vert
        else:
            vert = self.head
            self.head = self.head.Next
            self.head.Prev = None
            return vert


def list2str(L):
    """ 将列表转化为字符串的函数, 用逗号分隔各项 """
    st = ''
    for s in L:
        st += s+', '
    st = st[:-2]
    return st


if __name__ == "__main__":

    # 建图
    with open('Film.json', 'r') as f:
        DATA = json.load(f)
    G = Graph(DATA)

    # 设置输出文件, 输出markdown文件, 直接导入到报告中
    output = open('output.md', 'w')

    """
    Problem 1,2
    """
    output.write('### 连通分支\n')
    L = G.cnct_comp
    numBranch = len(L)
    output.write('共%d个连通分支.\n' % numBranch)
    NumList = []
    diameterList = []
    StarList = []
    output.write('|序号|演员个数|电影类别|直径|\n')
    output.write('|---|-----|----|---|\n')
    for i in range(20):
        Num = len(L[i])
        NumList.append(Num)
        diameter = 0
        if i >= 1:
            diameter = G.diameter(i)
        diameterList.append(diameter)
        films = G.films_in_cont(i)
        StarList.append(np.mean([f.star for f in films.values()]))
        type = {}
        for f in films.values():
            for t in f.type:
                if t in type:
                    type[t] += 1
                else:
                    type[t] = 1
        types = sorted(type.items(), key=lambda dict: dict[1], reverse=True)
        if len(types) >= 3:
            Type = [types[i][0] for i in range(3)]
        else:
            Type = [t[0] for t in types]
        output.write('|'+str(i+1)+'|'+str(Num)+'|' +
                     list2str(Type)+'|'+str(diameter)+'|\n')
    output.write('|...|...|...|...|\n')
    NumList.append(0)
    diameterList.append(0)
    StarList.append(0)
    for i in range(numBranch-20, numBranch):
        Num = len(L[i])
        NumList.append(Num)
        diameter = G.diameter(i)
        diameterList.append(diameter)
        films = G.films_in_cont(i)
        StarList.append(np.mean([f.star for f in films.values()]))
        type = {}
        for f in films.values():
            for t in f.type:
                if t in type:
                    type[t] += 1
                else:
                    type[t] = 1
        types = sorted(type.items(), key=lambda dict: dict[1], reverse=True)
        if len(types) >= 3:
            Type = [types[i][0] for i in range(3)]
        else:
            Type = [t[0] for t in types]
        output.write('|'+str(i+1)+'|'+str(Num)+'|' +
                     list2str(Type)+'|'+str(diameter)+'|\n')
    """
    Problem 3
    """
    output.write('### 画图\n')

    fig, ax = plt.subplots(4, gridspec_kw={'height_ratios':[2,5,7,7]}, sharex=True, figsize=(10, 10))
    plt.xticks(rotation=45)
    index = list(range(1,21))+['']+list(range(numBranch-19, numBranch+1))
    x = np.arange(len(index))
    N = np.array(NumList)
    R = np.array(diameterList)
    S = np.array(StarList)
    ax[0].bar(x, N)
    ax[1].bar(x, N)
    ax[0].set_ylim(84680,84700)
    ax[1].set_ylim(0,50)
    ax[0].spines['bottom'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[0].tick_params(length=0)
    d = .015
    kwargs = dict(transform=ax[0].transAxes, color='k', clip_on=False)
    ax[0].plot((-d, +d), (-5*d, +5*d), **kwargs)
    ax[0].plot((1 - d, 1 + d), (-5*d, +5*d), **kwargs)
    kwargs.update(transform=ax[1].transAxes)
    ax[1].plot((-d, +d), (1 - 2*d, 1 + 2*d), **kwargs)
    ax[1].plot((1 - d, 1 + d), (1 - 2*d, 1 + 2*d), **kwargs)
    ax[2].bar(x, R)
    ax[3].bar(x, S, tick_label=index)
    ax[0].set_title('Scales of the first and last 20 connected components')
    ax[2].set_title('Diameters of the first and last 20 connected components')
    ax[3].set_title('Average stars of the first and last 20 connected components')
    fig.savefig('2.png')
    output.write('@import "2.png"\n')

    """
    Problem 4,5
    """
    output.write('### 周星驰\n')
    zhou = G.vertlist['周星驰']
    films_by_zhou = zhou.films
    starzhou = np.mean([f.star for f in films_by_zhou.values()])
    output.write('周星驰出演电影的平均星级为%.2f.\n' % starzhou)
    coactors = list(zhou.getConnections())
    output.write('周星驰和他的的共同出演者一共%d人.\n' % (len(coactors)+1))
    coactors.append(zhou)
    all_films = {}
    for actor in coactors:
        for f in actor.films:
            all_films[f] = actor.films[f]
    number = len(all_films)
    star = np.mean([all_films[f].star for f in all_films])
    type = {}
    for f in all_films:
        film = all_films[f]
        for t in film.type:
            if t in type:
                type[t] += 1
            else:
                type[t] = 1
    types = sorted(type.items(), key=lambda dict: dict[1], reverse=True)

    output.write('周星驰和他的共同出演者共演出了%d部电影, 所出演电影的平均星级为%.2f, 电影所属类别前三名为' % (
        number, star)+list2str([types[i][0] for i in range(3)])+'.\n')
    """ ========================================================= """
    # print('下面列出周星驰和共同出演者所演电影的统计信息')
    # table = "{0:10}\t{1:^10}\t{2:10}\t{3:20}"
    # coactors.append(zhou)
    # print(table.format('演员姓名', '电影数目', '平均星级', '类型'))
    # for actor in coactors:
    #     name = actor.actor
    #     films = actor.films
    #     num_of_film = len(films)
    #     star_of_film = np.mean([f.star for f in films])
    #     type = {}
    #     for f in films:
    #         for t in f.type:
    #             if t in type:
    #                 type[t] += 1
    #             else:
    #                 type[t] = 1
    #     types = sorted(type.items(), key=lambda dict: dict[1], reverse=True)
    #     if len(types) >= 3:
    #         Type = [types[i][0] for i in range(3)]
    #     else:
    #         Type = [t[0] for t in types]
    #     print(table.format(name, num_of_film, '%.2f' % star_of_film, str(Type)))
    # print('### 探究出演电影数与出演电影平均星级的关系')
    # D={}
    # for actor in G.vertlist:
    #     films=G.vertlist[actor].films
    #     # coactors=G.vertlist[actor].getConnections()
    #     # n=len(coactors)
    #     n=0
    #     if actor!='':
    #         n=len(films)
    #     if n>250:
    #         print(actor)
    #     star=np.mean([f.star for f in films.values()])
    #     if n in D:
    #         D[n].append(star)
    #     else:
    #         D[n]=[star]
    # for n in D:
    #     D[n]=np.mean(D[n])
    # D=sorted(D.items(), key=lambda dict: dict[0])
    # x=[item[0] for item in D]
    # y=[item[1] for item in D]
    # plt.plot(x,y,'.')
    # plt.show()
    
