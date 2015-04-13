# -*- coding: utf-8 -*-
__author__ = 'tan'

import os
import logging

logname = os.path.basename(__file__) + ".log"
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    filename=logname)

import copy


def getScore(scores):
    '''

    :param scores:
    :return:
    '''
    old = copy.copy(scores)

    max_id = scores.index(max(scores))
    scores[max_id] = -100
    second_id = scores.index(max(scores))
    scores[second_id] = -100
    third_id = scores.index(max(scores))
    scores[third_id] = -100
    four_id = scores.index(max(scores))

    res = [3] * len(scores)
    # print(old)
    # if old[max_id] - old[second_id] < 0.001 and old[second_id] - old[third_id] > 0.1:
    #     res[max_id] = 1
    #     res[second_id] = 1
    #     res[third_id] = 2
    # elif old[second_id] - old[third_id] < 0.001 and old[third_id] - old[four_id] > 0.1:
    #     res[max_id] = 1
    #     res[second_id] = res[third_id] = 2
    # else:
    res[max_id] = 1
    res[second_id] = 2

    return res

def get_result(inname):
    pred = []
    f = open(inname, "rb")
    lines = f.readlines()
    f.close()

    tmp_id = ""
    scores = []
    for i in xrange(0, len(lines)):
        line = lines[i]
        line = line.split()
        _id = line[0]
        score = float(line[2])

        if tmp_id == "":
            tmp_id = _id
            scores.append(score)
        elif tmp_id == _id:
            scores.append(score)

        if tmp_id != _id or i == (len(lines) - 1):
            res = getScore(scores)
            pred.extend(res)

            tmp_id = _id
            scores = []
            scores.append(score)


    print(len(pred))

    with open("tmp.result.txt", "wb") as ff:
        for p in pred:
            ff.write(str(p)+"\n")




