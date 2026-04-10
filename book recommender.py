from tkinter import*
import pandas as pd

books=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\Books.csv")
books.info()

print(books.isna().sum())
books["No_description"]=books["description"]=="No description available"
books.info()

print(books["No_description"].sum())

#506 books have no description. how to remove them?

def show():
    select=lb.curselection()

    print(select)
    print(lb.get(select))

window=Tk()
window.geometry("500x500")
window.title("book recommender")

sb=Scrollbar(window,width=30)

lb=Listbox(window,yscrollcommand=sb.set)
lb.pack()
sb.pack(side=LEFT,fill=Y)
sb.config(command=lb.yview)

lab=Label(window,text="Select your favourite book",font=("comicsans",24,"bold"))
lab.pack()

for i, title in enumerate(books["title"]):
    lb.insert(i, title)

b=Button(window,text="go",command=show)
b.pack()




window.mainloop()