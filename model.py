# -*- coding: utf-8 -*-
__author__ = 'tan'

class Answer:
    def __init__(self):
        self.id = None
        self.gold = None
        self.body = None
        pass

    def setproperty(self, cid, gold, body):
        self.id = cid
        self.gold = gold
        # self.body = unicode(body).encode("utf-8")
        self.body = body

    def __str__(self):
        return unicode("[ id: "+self.id+" gold: "+self.gold+" body: "+self.body+" ]\n").encode("utf-8")
        # return "[ id: "+self.id+" gold: "+self.gold+" content: "+self.body+" ]\n"


class Question:
    def __init__(self):
        self.id = None
        self.category = None
        self.date = None
        self.answers = None
        self.body = None
        self.subject = None

    def setproperty(self, qid, category, qdate):
        self.id = qid
        # self.category = unicode(category).encode("utf-8")
        self.category = category
        self.date = qdate

    def set_body(self, subject, body):
        # self.subject = unicode(subject).encode("utf-8")
        # self.body = unicode(body).encode("utf-8")
        self.subject = subject
        self.body = body

    def add_answer(self, answers):
        self.answers = answers

    def __str__(self):
        tmpstr = unicode("[ id:"+self.id+" body: "+self.body).encode("utf-8")
        # tmpstr = "[ id:"+self.id+" body: "+self.body
        for ans in self.answers:
            tmpstr += ans.__str__()
        return tmpstr + " ]\n"