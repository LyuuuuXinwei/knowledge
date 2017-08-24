"""
1.pythonic:
print('hello,%s'%('tom',))不如print('hello,%(name)s%{'name':'tom'})

2.变量名和解决的问题域一致，为了让程序更容易理解，不要害怕过长的变量名

3.flask源码很pythonic

4.PEP8：代码风格指南，布局注释命名规范
pep8 filename.py
其他编程规范：google python style guide--pychecker
pyflakes集成到VIM中

5.给外界可以使用的函数和方法添加文档注释，说清功能，参数，返回值，有可能的异常说明，复杂的内部函数方法也需要注释,可读性可维护性
def func_name(parameter1,parameter2):
    '''disdribe what this function does.
        Args:
            parameter1:
            parameter2:
        Returns:
            return type return value
    '''
    function body

6.文件头：copyright声明，模块描述，作者信息，变更记录

        FileName:xxx.py
        Description:test

        Author:name
        Change Activity:list

7.使用适当空行使代码布局更合理

8.函数设计避免一个函数过长也要避免设计冗余函数嵌套深度过大，循环层数在3层内，考虑向下兼容，设置默认参数，参数就是接口

9.常量集中在一个文件，易于维护

10.python -O asserttest.py禁用断言；如果Python本身能处理就不要用断言

11.python交换 x,y=y,x

12.lazy evaluation：善用生成器，多想多优化改进

13.转换为浮点再做除法

14.eval('xxxx')将xxxx作为python表达式返回计算结果，eval如果读取了用户输入并执行，在web环境下很危险

15.获取序列迭代的索引和值enumerate，字典不适合
for i,e in enumerate(list):
    print('index:',i,'element:',e)

16.is对象标识符，同对象TRUE ==

17.unicode转换格式UTF

18.构建合理的包层次管理module

11.节制使用from import

12.else语法糖：上个结构未被自然执行，break也算未自然执行
如未break 执行else，如未except 则else
try except else finally

13string规模较大时''.join比+好

14.format比%更推荐作为格式化字符串的方式
format可以先构造出字典或元组，再参数传递元组，顺序不用固定，使用位置符/名称
Weather={('Monday':'rain'),('tuesday':'sunny')}
f='Today is {0[0]}.and weather is{1[1]}.'.format
for i in map(f,Weather):
print(i)

15.str()多面对用户repr()面对调试，大体相似

16.编程两件事：处理字符，数值。对商业应用来说，处理字符串的代码占了8成

17.排序算法时间复杂度,
sort()一般用于列表
sorted()用于任何可迭代的对象 sorted(dict.iteritems(),key=lambda (k,v): operator.itemgetter((1)(v))

18.elementtree解析XML不用DOM和SAX，性能好速度快

19.traceback模块：抽取打印栈跟踪信息
"""

