#!/usr/bin/env python3
# Author : Saikat Chakraborty (saikatc@cs.columbia.edu)
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Basic tokenizer that splits text into alpha-numeric tokens and
non-whitespace tokens.
"""

import logging
from tokenizer import Tokens, Tokenizer
import re

logger = logging.getLogger(__name__)
sep = '\t'

def tokenize_with_camel_case(token):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', token)
    return [m.group(0) for m in matches]


def tokenize_with_snake_case(token):
    return token.split('_')

def isNUM(z):
    try:
        z=int(z)
        return isinstance(z,int)
    except ValueError:
        return False

def split_some_token(cc): # split alNUMal
    nc=[]
    for c in cc: nc+=re.findall(r'(\d+|\D+)',c)
    return nc

def merge_some_token(cc): # merge pX p100
    nc=[]
    pass_next=False
    for c in range(len(cc)):
        if pass_next:
            pass_next=False
            continue
        if len(cc[c])==1 and c+1<len(cc):
            if (('a'<=cc[c]<='z' and ((len(cc[c+1])==1 and 'A'<=cc[c+1]<='Z') or isNUM(cc[c+1]))) 
                or ('A'<=cc[c]<='Z' and isNUM(cc[c+1]))):
                nc.append(cc[c]+cc[c+1])
                pass_next=True
            else:
                nc.append(cc[c])
        else:
            nc.append(cc[c])
    return nc

class CodeTokenizer(Tokenizer):
    def __init__(self, camel_case=True, snake_case=True, **kwargs):
        """
        Args:
            camel_case: Boolean denoting whether CamelCase split is desired
            snake_case: Boolean denoting whether snake_case split is desired
            annotators: None or empty set (only tokenizes).
        """
        super(CodeTokenizer, self).__init__()
        self.snake_case = snake_case
        self.camel_case = camel_case
        assert self.snake_case or self.camel_case, \
            'To call CodeIdentifierTokenizer at least one of camel_case or ' \
            'snake_case flag has to be turned on in the initializer'
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        tokens = text.split()
        snake_case_tokenized = []
        if self.snake_case:
            for token in tokens:
                snake_case_tokenized.extend(tokenize_with_snake_case(token))
        else:
            snake_case_tokenized = tokens
        camel_case_tokenized = []
        if self.camel_case:
            for token in snake_case_tokenized:
                cc=tokenize_with_camel_case(token)
                cc=split_some_token(cc)
                cc=merge_some_token(cc)
                camel_case_tokenized.extend(cc)
        else:
            camel_case_tokenized = snake_case_tokenized
        data = []
        for token in camel_case_tokenized:
            data.append((token, token, token))

        ws=Tokens(data, self.annotators).words()
        # w=''
        # for i in range(len(ws)): 
        #     w+=ws[i]
        #     if i!=len(ws)-1: w+=sep
        return ws


if __name__=='__main__':
    """
    单个字母后面加数字不拆 a0 -> a0 , a00 -> a00, 这条优先于第四、五条 i18n -> i18 n
    一个小写一个大写不拆 pX -> pX 
    多字母要拆 aaa0 -> aaa 0 前面就算只有两个字母也要拆
    字母数字字母要拆  aa0a -> aa 0 a 后面是小写也拆
    前数字后字母要拆 15MIN -> 15 MIN
    
    补丁：
    扫描分词结果
    先找单个小写字母，如果后面是单个大写，则拼起来(下划线分的就不拼)
    细分割：拆开数字字母混合词，拆完后如果发现数字前是单个字母，则拼上去
    """
    T=CodeTokenizer()
    s='nA'#'decodeBase256Segment'
    q=T.tokenize(s)
    print(q)
    
    
    