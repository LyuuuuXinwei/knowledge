m=1 if b else 2 # 赋值简化写法

# 变量名称区分大小写

module.__name__ # 被导入：module，本身：__main__

p=itertools.permutation([1,2,3,4],3)
next(p) # 元素互相排列组合的方式
itertools.product('ABC','123') 两个序列笛卡尔积
itertools.groupby('序列',key) key 为name,len等
itertools.chain()

random.seed方法的作用是给随机数对象一个种子值，用于产生随机序列。
对于同一个种子值的输入，之后产生的随机数序列也一样。
通常是把时间秒数等变化值作为种子值，达到每次运行产生的随机系列都不一样。
random.seed(datetime.datetime.now())