import cv2
from tkinter import *
import numpy as np
import pandas as pd

df=pd.read_csv("PerpData.csv")

root = Tk()
root.configure(background='navyblue')

# entry variables
name = StringVar()
percent = StringVar()
backlog = StringVar()
intern = StringVar()
firstround = StringVar()
commskills = StringVar()

# photo

# Heading
w2 = Label(root, justify=LEFT, text="Nucleus Computers Ltd.", fg="white", bg="navyblue")
w2.config(font=("Elephant", 20))
w2.grid(row=0, column=0, columnspan=2, padx=200)

# labels
nameLb = Label(root, text="Name", fg="white", bg="navyblue")
nameLb.grid(row=1, column=0, pady=10, sticky=W)

percentageLb = Label(root, text="PERCENTAGE", fg="white", bg="navyblue")
percentageLb.grid(row=2, column=0, pady=10, sticky=W)

backlogLb = Label(root, text="BACKLOG", fg="white", bg="navyblue")
backlogLb.grid(row=3, column=0, pady=10, sticky=W)

internLb = Label(root, text="INTERNSHIP", fg="white", bg="navyblue")
internLb.grid(row=4, column=0, pady=10, sticky=W)

frLb = Label(root, text="FIRST ROUND", fg="white", bg="navyblue")
frLb.grid(row=5, column=0, pady=10, sticky=W)


cmskillsLb = Label(root, text="COMMUNICATION SKILLS", fg="white", bg="navyblue")
cmskillsLb.grid(row=6, column=0, pady=10, sticky=W)

resultLb = Label(root, text="RESULT", fg="white", bg="navyblue")
resultLb.grid(row=7, column=0, pady=10)

destreeLb = Label(root, text="DecisionTree", fg="white", bg="navyblue")
destreeLb.grid(row=8, column=0, pady=10, sticky=W)

ranfLb = Label(root, text="RandomForest", fg="white", bg="navyblue")
ranfLb.grid(row=9, column=0, pady=10, sticky=W)

logrLb = Label(root, text="LogisticRegression", fg="white", bg="navyblue")
logrLb.grid(row=10, column=0, pady=10, sticky=W)

# entries

nameEn = Entry(root, textvariable=name)
nameEn.grid(row=1, column=2)

perEn = Entry(root, textvariable=percent)
perEn.grid(row=2, column=2)

bklEn = Entry(root, textvariable=backlog)
bklEn.grid(row=3, column=2)

intEn = Entry(root, textvariable=intern)
intEn.grid(row=4, column=2)

frEn = Entry(root, textvariable=firstround)
frEn.grid(row=5, column=2)

comsEn = Entry(root, textvariable=commskills)
comsEn.grid(row=6, column=2)

X = df[["PERCENTAGE","BACKLOG","INTERNSHIP","FIRSTROUND","COMMUNICATIONSKILLLS"]]
y = df[["Hired"]]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=26)

model = LinearRegression()
model.fit(X_train,y_train)

# inputTest = df[perEn.get(), bklEn.get(),intEn.get(),
#              frEn.get(),comsEn.get()]

# inputTest = df["PERCENTAGE":perEn.get(),"BACKLOG": bklEn.get(),"INTERNSHIP":intEn.get(),
#              "FIRSTROUND":frEn.get(),"COMMUNICATIONSKILLS":comsEn.get()]

# inputTest = np.array([float(perEn.get()), float(bklEn.get()),float(intEn.get()),float(frEn.get()),float(comsEn.get())]).reshape(1,-1)
# predicted = model.predict(inputTest)
# inputTest = np.array([perEn.get(), bklEn.get(),intEn.get(),frEn.get(),comsEn.get()]).reshape(1,-1)

# inputTest = np.array([80,1,1,88,90]).reshape(1,-1)
# predicted = model.predict(inputTest)
# print(predicted)

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def decisiontree():
    df=pd.read_csv('PerpData.csv')
    df['BACKLOG'] = (df.BACKLOG>0).astype(int)

    features = list(df.columns[1:6])
    # features

    y = df["Hired"]
    x = df[features]
    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(x,y)         # train the above model

    predicted = clf3.predict([[perEn.get(),bklEn.get(),intEn.get(),frEn.get(),comsEn.get()]])
    if(predicted == 1):
        # t1.insert(END, "Hired")
        t1.delete(1.0,END)
        t1.insert(END, "Hired")
    else:
        # t1.insert(END, "Not Hired")
        t1.delete(1.0,END)
        t1.insert(END, "Not Hired")


def RandomForest():
    df=pd.read_csv('PerpData.csv')
    df['BACKLOG'] = (df.BACKLOG>0).astype(int)

    features = list(df.columns[1:6])
    y = df["Hired"]
    x = df[features]


    clf4 = RandomForestClassifier(n_estimators=10)        #no. of trees
    clf4 = clf4.fit(x,y)



    # print(clf4.predict([[84,0,3,90,80]]))
    predicted = clf4.predict([[perEn.get(),bklEn.get(),intEn.get(),frEn.get(),comsEn.get()]])



    if(predicted == 1):
        # t1.insert(END, "Hired")
        t2.delete(1.0,END)
        t2.insert(END, "Hired")
    else:
        # t1.insert(END, "Not Hired")
        t2.delete(1.0,END)
        t2.insert(END, "Not Hired")
#giving values to all the fields
# perEn.insert(END, df["PERCENTAGE"].get(df["NAME":nameEn.get()])

# buttons

def logisticRegression():
    df=pd.read_csv('PerpData.csv')
    df['BACKLOG'] = (df.BACKLOG>0).astype(int)
    # df=df.columns[1:6].astype(int)

    features = list(df.columns[1:6])
    y = df["Hired"]
    x = df[features]


    clf5 = LogisticRegression()        #no. of trees
    clf5 = clf5.fit(x,y)



    # print(clf4.predict([[84,0,3,90,80]]))
    predicted = clf5.predict(np.array([float(perEn.get()),float(bklEn.get()),float(intEn.get()),float(frEn.get()),float(comsEn.get())]).reshape(1,-1))



    if(predicted == 1):
        # t1.insert(END, "Hired")
        t3.delete(1.0,END)
        t3.insert(END, "Hired")
    else:
        # t1.insert(END, "Not Hired")
        t3.delete(1.0,END)
        t3.insert(END, "Not Hired")

dst = Button(root, text="DecisionTree", command=decisiontree,bg="blue",fg="white")
dst.grid(row=2, column=3,padx=10)

rnf = Button(root, text="Randomforest", command=RandomForest,bg="blue",fg="white")
rnf.grid(row=3, column=3,padx=10)

dst = Button(root, text="logisticRegression", command=logisticRegression,bg="blue",fg="white")
dst.grid(row=4, column=3,padx=10)

#textfileds
t1 = Text(root, height=1, width=20,bg="navyblue",fg="white")
t1.grid(row=8, column=1)

t2 = Text(root, height=1, width=20,bg="navyblue",fg="white")
t2.grid(row=9, column=1)

t3 = Text(root, height=1, width=20,bg="navyblue",fg="white")
t3.grid(row=10, column=1)

root.mainloop()
