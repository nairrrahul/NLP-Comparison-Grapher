import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

window = tk.Tk()
window.title("NLP Comparison Grapher")
window.iconbitmap('favicon.ico')

vecs = [TfidfVectorizer(), CountVectorizer()]
vecs_2 = [TfidfVectorizer(), CountVectorizer()]
vecs_str = ["TfidfVectorizer()", "CountVectorizer()"]
clasfs = [LinearSVC(), KNeighborsClassifier()]
clasfs_2 = [LinearSVC(), KNeighborsClassifier()]
clasfs_str = ["LinearSVC()", "KNeighborsClassifier()"]
choices = [0, 1, 0, 1]

main_sheet = pd.read_csv('QuestionAnswerPairClean.csv')[['question_format', 'answer_format', 'label']].\
    dropna(subset=['label'])

title = tk.Label(text="NLP Comparison Grapher", font=("Century Gothic", 15))
title.grid(column=0, row=0, columnspan=4)

sub_1 = tk.Label(text="Classifiers and Vectorizers", font=("Century Gothic", 12))
sub_1.grid(column=0, row=1, columnspan=4, padx=10, pady=10)

v1_title = tk.Label(text="Vectorizer 1")
v2_title = tk.Label(text="Vectorizer 2")
c1_title = tk.Label(text="Classifier 1")
c2_title = tk.Label(text="Classifier 2")

v1_title.grid(column=0, row=2)
v2_title.grid(column=1, row=2)
c1_title.grid(column=2, row=2)
c2_title.grid(column=3, row=2)

m_v = tk.StringVar(window)
m_v.set(vecs[0])
m_v2 = tk.StringVar(window)
m_v2.set(vecs[1])
vec1 = tk.OptionMenu(window, m_v, *vecs)
vec2 = tk.OptionMenu(window, m_v2, *vecs)
vec1.grid(column=0, row=3)
vec2.grid(column=1, row=3)

c_v = tk.StringVar(window)
c_v.set(clasfs[0])
c_v2 = tk.StringVar(window)
c_v2.set(clasfs[1])
cl1 = tk.OptionMenu(window, c_v, *clasfs)
cl2 = tk.OptionMenu(window, c_v2, *clasfs)
cl1.grid(column=2, row=3)
cl2.grid(column=3, row=3)

sub_2 = tk.Label(text="Extra Customizations", font=("Century Gothic", 12))
sub_2.grid(column=0, row=4, columnspan=4, padx=10, pady=10)

x1_title = tk.Label(text="Cross-Validations")
x2_title = tk.Label(text="C₁ (LinearSVC)")
x3_title = tk.Label(text="n_neighbors₁ (KNN)")
x4_title = tk.Label(text="Number of Runs")

x1_title.grid(column=0, row=5)
x2_title.grid(column=1, row=5)
x3_title.grid(column=2, row=5)
x4_title.grid(column=3, row=5)

y1_entry = tk.Entry()
y1_entry.insert(0, 1)
y2_entry = tk.Entry()
y2_entry.insert(0, 1)
y3_entry = tk.Entry()
y3_entry.insert(0, 1)
y4_entry = tk.Entry()
y4_entry.insert(0, 1)

y1_entry.grid(column=0, row=6)
y2_entry.grid(column=1, row=6)
y3_entry.grid(column=2, row=6)
y4_entry.grid(column=3, row=6)

z1_title = tk.Label(text="C₂ (LinearSVC)")
z2_title = tk.Label(text="n_neighbors₂ (KNN)")

z1_title.grid(column=1, row=7)
z2_title.grid(column=2, row=7)

z1_entry = tk.Entry()
z1_entry.insert(0, 1)
z2_entry = tk.Entry()
z2_entry.insert(0, 1)

z1_entry.grid(column=1, row=8)
z2_entry.grid(column=2, row=8)


# UPDATING CHOICES FOR SELECTIONS
def update_choices():
    choices[0], choices[1], choices[2], choices[3] = \
        vecs_str.index(m_v.get()), vecs_str.index(m_v2.get()), \
        clasfs_str.index(c_v.get()), clasfs_str.index(c_v2.get())


# CHECKING TYPE and NUMBER OF CUSTOMIZATIONS
def check_errors():
    update_choices()
    try:
        int(y1_entry.get())
        try:
            int(y3_entry.get())
            try:
                float(y2_entry.get())
                if int(y1_entry.get()) < 1 or int(y3_entry.get()) < 1:
                    error_text.configure(text="Cross-Validations or n_neighbors has to be >= 1")
                elif float(y2_entry.get()) <= 0:
                    error_text.configure(text="C-Value has to be > 0")
                else:
                    error_text.configure(text="")
            except ValueError:
                error_text.configure(text="ValueError: C needs to be 'float'")
        except ValueError:
            error_text.configure(text="ValueError: n_neighbors needs to be 'int'")
    except ValueError:
        error_text.configure(text="ValueError: Cross-Variations needs to be 'int'")


# RUN ALGORITHM
def nlp_values(vec_1, vec_2, clasf_1, clasf_2, n_ns, c_val, n_ns2, c_val2, cross_num):
    check_errors()
    if len(error_text.cget("text")) > 0:
        return -1, -1
    avg_1 = 0
    avg_2 = 0
    for i in range(0, cross_num):
        vector_1 = vec_1
        vector_2 = vec_2

        train, test = train_test_split(main_sheet, test_size=0.25)

        tr_1 = train['question_format']
        te_1 = test['question_format']
        tr_2 = train['label']
        te_2 = test['label']

        val_train, val_test = vector_1.fit_transform(tr_1), vector_1.transform(te_1)
        val_train2, val_test2 = vector_2.fit_transform(tr_1), vector_2.transform(te_1)

        if clasfs.index(clasf_1) == 0:
            n1 = clasf_1
            n1.C = c_val
        else:
            n1 = clasf_1
            n1.n_neighbors = n_ns

        if clasfs_2.index(clasf_2) == 0:
            n2 = clasf_2
            n2.C = c_val2
        else:
            n2 = clasf_2
            n2.n_neighbors = n_ns2

        n1.fit(val_train, tr_2)
        n2.fit(val_train2, tr_2)

        avg_1 += np.mean(n1.predict(val_test) == te_2)
        avg_2 += np.mean(n2.predict(val_test2) == te_2)

    return avg_1/cross_num, avg_2/cross_num


# RUN GRAPHER
def grapher():
    res_1 = []
    res_2 = []
    runs = []
    final_graph_notif.configure(text="")
    error_text.configure(text="")
    try:
        int(y4_entry.get())
        if int(y4_entry.get()) <= 0:
            error_text.configure(text="Number of Runs needs to be >= 1")
        if len(error_text.cget("text")) > 0:
            final_graph_notif.configure(text="Graph cannot be generated")
        else:
            final_graph_notif.configure(text="Generating Graph...")
            error_text.configure(text="")
            for i in range(1, int(y4_entry.get()) + 1):
                x1, x2 = nlp_values(vecs[choices[0]], vecs_2[choices[1]],
                                    clasfs[choices[2]], clasfs_2[choices[3]], int(y3_entry.get()),
                        float(y2_entry.get()), int(z2_entry.get()),
                                    float(z1_entry.get()), int(y1_entry.get()))
                res_1.append(x1)
                res_2.append(x2)
                runs.append(i)

            fig = plt.figure(figsize=(5, 4))
            graph = fig.add_subplot(111)
            str1, str2 = vecs_str[choices[0]][0: len(vecs_str[choices[0]]) - 2] + ", " + \
                         clasfs_str[choices[2]][0: len(clasfs_str[choices[2]]) - 2], \
                         vecs_str[choices[1]][0: len(vecs_str[choices[1]]) - 2] + ", " + \
                         clasfs_str[choices[3]][0: len(clasfs_str[choices[3]]) - 2]
            str1 += " (C= " + y2_entry.get() + ")" if choices[2] == 0 else " (neighbors= " + y3_entry.get() + ")"
            str2 += " (C= " + z1_entry.get() + ")" if choices[3] == 0 else " (neighbors =" + z2_entry.get() + ")"
            graph.plot(runs, res_1, label=str1)
            graph.scatter(runs, res_1)
            graph.plot(runs, res_2, label=str2)
            graph.scatter(runs, res_2)
            graph.set_xlabel("Runs")
            graph.set_ylabel("Accuracy")
            plt.axis((0, int(y4_entry.get()) + 1, 0.15, 0.9))
            graph.legend()

            graph_canvas = FigureCanvasTkAgg(fig, master=window)
            graph_canvas.draw()
            graph_canvas.get_tk_widget().grid(column=0, row=12, columnspan=4)

            final_graph_notif.configure(text="Graph Generated")
    except ValueError:
        error_text.configure(text="Number of Runs needs to be an integer")


get_button = tk.Button(text="Generate", fg="white", bg="black", command=grapher)
error_text = tk.Label(fg="red")
final_graph_notif = tk.Label()

get_button.grid(column=1, row=9, columnspan=2, padx=10, pady=10)
error_text.grid(column=0, row=10, columnspan=4)
final_graph_notif.grid(column=0, row=11, columnspan=4)


window.mainloop()
