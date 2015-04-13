# -*- coding: utf-8 -*-
__author__ = 'tan'


from bs4 import BeautifulSoup as bs
from model import Question, Answer
import cPickle


'''
<Question QID = "20831" QCATEGORY = "فقه العبادات > الطهارة" QDATE = "2002-13-08">
	<QSubject>حكم تشذيب اللحية</QSubject>
	<QBody>أنا شخص لدي لحية خفيفة وهي غير متساوية. ما الحكم إذا خففتها وساويتها على هيئة ما يعرف بالعارض؟ أفيدوني جزاكم الله خيراً.</QBody>
	<Answer CID = "41673" CGOLD = "direct">الحمد لله والصلاة والسلام على رسول الله وعلى آله وصحبه أما بعد: فخفة اللحية وعدم تساوي أطرافها لا يبيح لك أخذ ما دون القبضة منها لحرمة ذلك، إلا إذا كانت متباينة الأطراف تبايناً فاحشاً مشوهاً للخلقة فيقتصر على قص الزائد الذي حصل به التشويه فقط كما هو مبين في الفتوى رقم:  والله أعلم.</Answer>
	<Answer CID = "21282" CGOLD = "related">الحمد لله والصلاة والسلام على رسول الله وبعد: فإذا كان الأمر قد وصل بك إلى الحالة التي ذكرتها، ولم يمكنك أن تجد علاجا ناجعا في إعادة إنبات شعر اللحية إلى طبيعته، وكان في بقاء هذا الشق من شعر وجهك تشويه فادح للخلقة، فلا مانع إن شاء الله من تشذيبه ليماثل شعر لحيتك في الشق الآخر، فإذا عاد شعر لحيتك إلى ما كان عليه فيجب الكف عن تشذيبها والأخذ منها لضرورة تقدر بقدرها. والله أعلم</Answer>
	<Answer CID = "157620" CGOLD = "irrelevant">فمجرد اختلاط نقود الفرع الإسلامي وأصله الربوي لا تأثير له لأن هذه الأموال المختطلة حلال وحرام ولأن الحرمة لا تتعلق بعين النقود، وإنما المعتبر هو مدى التزام الفرع الإسلامي بالضوابط الشرعية في معاملاته المالية وعدم تأثره بأصله الربوي كما بينا في الفتوى رقم: ، والفتوى رقم: . والله أعلم.</Answer>
	<Answer CID = "189787" CGOLD = "irrelevant">الحمد لله والصلاة والسلام على رسول الله وعلى آله وصحبه، أما بعد: فلا حرج عليك في الانتفاع بتلك العبوة التي اشتريتها من أولئك القوم وما دمت أقدمت على الشراء ناسيا لحكم المقاطعة التجارية فلا إثم عليك, لما روى ابن ماجه، وابن حبان، والدارقطني، والطبراني، والبيهقي والحاكم، أن النبي صلى الله عليه وسلم قال: رفع عن أمتي الخطأ والنسيان. حسنه النووي . وفي رواية: إن الله تجاوز لأمتي عن الخطأ والنسيان . متفق عليه. ولا يلزمك إتلاف العبوة أوالتخلص منها، بل إن إتلافها إفساد للمال، وللفائدة انظر الفتويين رقم :  ، ورقم :  . والله أعلم.</Answer>
	<Answer CID = "181271" CGOLD = "irrelevant">فلا إثم عليك ـ إن شاء الله ـ في العزوف عن الزواج ما دام ذلك لا يوقعك في الحرام، وانظري الفتوى رقم: . لكن ينبغي أن لا تستسلمي لهذا الخوف وتضيعي فرص الزواج، فاستعيني بالله عز وجل والتمسي علاج هذا الخوف عند أهل الاختصاص في علم النفس، ويمكنك التواصل مع قسم الاستشارات النفسية بموقعنا لمزيد من الفائدة. والله أعلم.</Answer>
</Question>
'''



from postagger import word_segment

def readFile(name):
    with open(name, 'rb') as f:
        content = f.read()
        data = bs(content, 'xml')
        root = data.root
        questions = []
        for node in root.findChildren('Question'):
            qid = node['QID']
            print("processing QID", qid)
            qcategory = node['QCATEGORY']
            qdate = node['QDATE']
            question = Question()
            question.setproperty(qid, qcategory, qdate)
            qsubject = word_segment(node.QSubject.get_text())
            qbody = word_segment(node.QBody.get_text())
            question.set_body(qsubject, qbody)

            answers = []
            for ch in node.findChildren('Answer'):
                cgold = ch['CGOLD']
                cid = ch['CID']
                answer = Answer()
                answer.setproperty(cid, cgold, word_segment(ch.get_text()))
                answers.append(answer)
            question.add_answer(answers)
            questions.append(question)
    return questions


def save_questions(name, outname):

    print("save into file %s " % outname)
    ques = readFile(name)
    # print(type(ques))
    cPickle.dump(ques, open(outname, 'wb'))

def merge_two_libsvm(one, two):
    from scipy.sparse import hstack
    new = hstack([one, two])
    return new


if __name__ == "__main__":
    # BASEDIR = ur'D:\百度云同步盘\Research\semeval2015\task3\arabic\semeval2015-task3-arabic-data\datasets'
    import os
    # train = BASEDIR + os.sep + 'QA-Arabic-train.xml'

    from config import train_questions_file_name, devel_questions_file_name, BASEDIR
    from config import train_file_name, devel_file_name
    questions = readFile(devel_file_name)
    print(len(questions))
    # for inx, que in enumerate(questions):
    #     # print(que)
    #     for ans in que.answers:
    #         print(ans.id)
    #         # print(unicode(ans).encode("utf-8"))
    #         print(ans.body)

    # cPickle.dump(questions, open("tmp.obj", 'wb'))

    # save_questions(train_file_name, BASEDIR + os.sep + train_questions_file_name)
    save_questions(devel_file_name, BASEDIR + os.sep + devel_questions_file_name)