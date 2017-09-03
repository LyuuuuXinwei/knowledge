import os,glob

[(os.stat(x).st_size,os.path.realpath(x)) for x in glob.glob('*.xml') if os.stat(x).st_size>6000]

{os.stat(x).st_size:os.path.realpath(x)) for x in glob.glob('*.xml') if os.stat(x).st_size>6000}

集合解析同

dict={}
{value:key for key,value in dict.items()}
# 交换键值对

生成器表达式取代列表解析可以节省内存CPU，惰性，传递给别的函数使用
'''外边改成圆括号即可'''

