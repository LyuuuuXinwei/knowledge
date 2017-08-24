__init__ # 类的构造函数，初始化，检验合法类的对象

instance.__class__ # 类名

生成器：yield的函数
迭代器：__inter__的类，__inter__被For时被调用，常用来迭代器的初始化，返回实现了__next__的对象，常为self
for直到StopIteration异常，不断调用__next__