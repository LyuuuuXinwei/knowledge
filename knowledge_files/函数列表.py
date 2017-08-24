

match_ # 正则检查需要被变复数的单词
apply_ # 以正则方式替换

import re
def match_sxz(noun):
    return re.search('[sxz]$', noun)
def apply_sxz(noun):
    return re.sub('$', 'es', noun)

'''rules函数列表'''
rules = ((match_sxz, apply_sxz),
    (match_h, apply_h),
    (match_y, apply_y),
    (match_default, apply_default)
    )
def plural(noun):
    for matches_rule, apply_rule in rules:
        if matches_rule(noun):
            return apply_rule(noun)

'''体会下高级语言额简洁'''
'''闭合，动态函数~'''
def build_match_and_apply_functions(pattern, search, replace):
    def matches_rule(word):
        return re.search(pattern, word)
    def apply_rule(word): ②
        return re.sub(search, replace, word)
    return(matches_rule, apply_rule)

'''构造一个四组参数的tuple，传入参数的构造整齐简洁'''
patterns=(('[sxz]$', '$', 'es'),('[^aeioudgkprt]h$', '$', 'es'),('(qu|[^aeiou])y$', 'y$', 'ies'),('$', '$', 's'))
'''for传递成组参数，结果简洁'''
rules = [build_match_and_apply_functions(pattern, search, replace) for (pattern, search, replace) in patterns]
