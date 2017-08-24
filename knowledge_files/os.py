# 所有导入模块必须添加到搜索路径

import os
os.getcwd() # 当前工作目录
os.chdir(path) # 改变工作目录,进入更深层目录可仅写example/

os.path # 操作文件名和目录名
os.path.join() # 无视斜杠问题链接path

(dirname,filename)=os.path.split()
(shortname,extension)=os.path.splitext(filename)

import glob
glob.glob('eg/*.xml') # （工作路径下）eg子目录下.xml文件
glob.glob(''*test*.py'') # 当前目录下名字里有test的.py文件

m=os.stat('filename.py') #返回一个包含多种文件原信息的对象
m.st_mtime # 最后修改时间的秒数计数
m.st_size #字节大小

os.path.realpath() #绝对路径，带盘符的