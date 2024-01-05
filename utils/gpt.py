#使用chatgpt生成sft数据
import openai
import os
import json
import random 
with open('prompt.txt','r',encoding='utf-8') as r:
    prompt=r.read()

# 书名、角色、情节列表，用于种子生成问题
books=['《射雕英雄传》','《神雕侠侣》','《天龙八部》']
characters={'《射雕英雄传》':['郭靖', '黄蓉', '杨康', '穆念慈', '黄药师', '欧阳锋','段智兴', '洪七公', '王重阳', '欧阳克', '周伯通'],
            '《神雕侠侣》':['杨过', '小龙女', '郭靖', '黄蓉', '郭襄', '程英', '陆无双'],
            '《天龙八部》':['段誉', '萧峰', '虚竹', '慕容复', '王语嫣', '阿朱', '阿紫', '段正淳', '木婉清']}
plots={'《射雕英雄传》':['比武招亲', '桃花岛奇遇', '襄阳保卫战', '九阳真经', '洪七公的武功传承', '华山论剑'],
       '《神雕侠侣》':['杨过断臂', '小龙女独闯全真教', '西毒北丐华山绝顶最终战'],
       '《天龙八部》':['少室山大战', '珍珑棋局', '段誉天龙寺学六脉神剑',  '丐帮大会', '聚贤庄大战','虚竹单挑鸠摩智','段誉失忆']}

# gpt代理 
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 运行请填入api_key
openai.api_key=''

# 种子列表
with open('seed_questions.txt','r',encoding='utf-8') as f:
    seeds = f.read().splitlines()

dic=[]
def get_response(text):# gpt获取答案
    
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=3500,
        messages=[
            {"role": "user", "content": f"{text}"},
        ]
    )
    # 将问题和答案都统一为萧峰，便于模型学习
    response = res.choices[0].message["content"].replace('乔峰','萧峰').splitlines()
    response=[line for line in response if line!=""]
    
    q=text.splitlines()
    for i in range(5):# 问答数据加入dic
        d={}
        d['Question']=q[i+5][3:]
        d['Answer']=response[i][3:]
        dic.append(d)

questions=[]
# 生成问题集
for book in books:
    for i in range(10):
        questions.append(seeds[i].format(book=book))

    for c1 in range(len(characters[book])):
        ch1=characters[book][c1]
        if((ch1=='郭靖' or ch1=='黄蓉') and book=='《神雕侠侣》'):
            questions.append(seeds[12].format(ch1=ch1,book=book))
        else:
            for i in range(10,18):
                questions.append(seeds[i].format(ch1=ch1,book=book))
        for c2 in range(c1+1,len(characters[book])):
            ch2=characters[book][c2]
            if(ch1=='郭靖' and ch2=='黄蓉' and book=='《神雕侠侣》'):
                continue
            for i in range(18,21):
                questions.append(seeds[i].format(ch1=ch1,ch2=ch2))
    
    for plot in plots[book]:
        for i in range(21,23):
            questions.append(seeds[i].format(book=book,plot=plot))

# 将问题打乱后再传给gpt
random.shuffle(questions)
# print(questions)
txt=''
for i in range(len(questions)):
    txt=txt+str(i%5+1)+'. '+questions[i]+'\n'
    if((i+1)%5==0):# 五个问题一组
        get_response(prompt+txt)
        txt=''

# 关系类问题将两个角色交换顺序，答案不变再存一次
for i in dic[:]:
    q=i['Question']
    po=-1
    if(q.find('是如何相遇')!=-1):
        po=q.find('是如何相遇')
    if(q.find('有什么故事')!=-1):
        po=q.find('有什么故事')
    if(q.find('之间有什么关系')!=-1):
        po=q.find('之间有什么关系')
    if(po==-1):
        continue
    q1=q[:po]
    ch=q1.split('和')
    q=q.replace(ch[0]+'和'+ch[1],ch[1]+'和'+ch[0])
    a=i['Answer'].replace(ch[0]+'和'+ch[1],ch[1]+'和'+ch[0])
    dic.append({'Question':q,'Answer':a})

# 再打乱一次，存入文件
random.shuffle(dic)
print(len(dic))
j=open('sft_data.jsonl','w',encoding='utf-8')
for i in dic:
    json.dump(i,j,ensure_ascii=False)
    j.write('\n')
j.close()
