# 节点：名称、权重、左右子节点
class Node(object):
    def __init__(self,name=None,value=None):
        self.name = name
        self.value = value
        self.left = None
        self.right = None

# 构造Huffman树
class HuffmanTree(object):
    def __init__(self,input):
        # 将所有节点名称和权重保存在列表中
        self.Leav = [Node(x[0],x[1]) for x in input]
        # 循环创建Huffman树，长度为1时直接作为根节点
        while len(self.Leav)!=1:
            # 从大到小排序
            self.Leav.sort(key=lambda node:node.value,reverse=True)
            # 创建父节点
            c = Node(value=(self.Leav[-1].value+self.Leav[-2].value))
            c.left = self.Leav[-1]
            c.right=self.Leav[-2]
            self.Leav = self.Leav[:-2]
            self.Leav.append(c)
        self.root = self.Leav[0]
        self.Buffer = list(range(10))
    def pre(self,tree,length):
        node = tree
        if (not node):
            return
        elif node.name:
            print(node.name + '    encoding:',end=''),
            for i in range(length):
                print (self.Buffer[i],end='')
            print()
            return
        self.Buffer[length]=0
        self.pre(node.left,length+1)
        self.Buffer[length]=1
        self.pre(node.right,length+1)

     # 生成哈夫曼编码
    def get_code(self):
        self.pre(self.root,0)
if __name__=='__main__':
    # 输入的是字符及其频数
    char_weights=[('a',6),('b',4),('c',10),('d',8),('f',12),('g',2)]
    tree=HuffmanTree(char_weights)
    tree.get_code()
