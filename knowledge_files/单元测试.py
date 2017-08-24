TDD测试驱动开发
import unittest,被测试函数
class Kownvalues(unittest.TestCase):#使该测试用例类称为unittest.TestCase的子类

    kownvalues=() #用例枚举表，边界用例，各种可能的输入

    def test_fuc(self):
        '''from_roman should fail with repeated pairs of numerals''' #在执行测试时会被打印出来，引发无误unittest返回OK
        self.assertEqual(a,b)
class Xss(unittest.TestCase):
    def test_func(self):

        self.assertRaises(TypeError,func,会引发错误的参数)

单元：一类测试一种文体,func_name必须以test开头

if __name__ == '__main__':
    unittest.main()

一旦习惯单元测试后，您可能会对自己曾在编程时不进行测试感到很奇怪
'''
unittest返回：
from_roman should fail with blank string ... ok
from_roman should fail with malformed antecedents ... ok
from_roman should fail with non‐string input ... ok
from_roman should fail with repeated pairs of numerals ... ok
Ran 12 tests in 0.203s
'''