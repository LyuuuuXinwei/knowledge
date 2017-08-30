

list[1:3] # 前面的数：第一个想要的元素 后面的数：第一个不想要的元素
list[:3] #
list[:] ==list

list+=['x']
list.append([1,2]) # 仅添加列表一项！！将列表加入列表
list.extend()
list.insert(0,,'dsss') #位置，内容

list.count() # 返回元素出现次数
'd'in list # true/false
list.index() # 元素首次出现位置

del list[1]
list.remove('s')
list.pop() # 弹出并删掉原末尾
list.pop(2) # 2位置

tuple 只能切片查询

list()
tuple()
set()
转

(x,y,z)=(1,2,3) # 元组多赋值

set={1,2}
set.add()
set.update({},{},[]) # 集合没有重复值
aet.remove()
set.discard() # remove一个不存在的值会报错discard不会
set.pop()
set.clear() # 清空
set1.union(set2) # 并
set1.intersection(set2) # 交
set1.difference(set2)

dict的值可以是任何数据类型

d = dict([('a', 1), ('b', 2), ('c', 3)]) d变为：{'a': 1, 'c': 3, 'b': 2}

dict.items()返回键值元组组成的列表
d={'1':'a','2':'b','3':'c'}
d.items()
[('1', 'a'), ('2', 'b'), ('3', 'c')]

None:nonetype,空值

a=[['user', 'pilgrim'], ['database', 'master'], ['password', 'PapayaWhip']]
dict=dict(a)
{'password': 'PapayaWhip', 'user': 'pilgrim', 'database': 'master'}

list-of-lists

zip()用法：多个列表结合
l=[1,12,123,1234,12345]
print([{}]*5) #[{}, {}, {}, {}, {}]
print([2,3]*5) #[2, 3, 2, 3, 2, 3, 2, 3, 2, 3]
print(dict(zip(l,[{}]*len(l)))) #{1: {}, 1234: {}, 123: {}, 12: {}, 12345: {}}
