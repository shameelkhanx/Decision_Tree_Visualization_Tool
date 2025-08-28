import streamlit as st 
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay




X,y=make_classification(n_samples=1000, n_features=2, n_informative=2,n_redundant=0,
 n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=0.5)


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=43)


    
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False
if "params" not in st.session_state:
    st.session_state.params = {}
    

option=st.sidebar.title("Parameters")
criteria =st.sidebar.selectbox("criterion :",["gini","entropy"])
max_depth = st.sidebar.number_input("MAX_DEPTH",min_value=1,max_value=100,value=3)
split = st.sidebar.selectbox("Splitter : ",["best","random"])
max_feature = st.sidebar.slider("MAX_FEATURE",min_value=1,max_value=2,value=1)
min_samples_leaf = st.sidebar.number_input("Min_Sample_leaf",min_value=1,max_value=100,value=3)
min_sample_split = st.sidebar.slider("Min_Sample_Split",min_value=1,max_value=500,value=50)

current_params = {
    "criterion": criteria,
    "max_depth": max_depth,
    "splitter": split,
    "max_feature": max_feature,
    "min_samples_leaf": min_samples_leaf,
    "min_sample_split": min_sample_split
}

if current_params != st.session_state.params:
    st.session_state.params = current_params
    st.session_state.button_clicked = False
   

    


def handle_button_click():
    st.session_state.button_clicked = True

st.sidebar.button("RUN ALGORITHUM",on_click=handle_button_click)


    
if not st.session_state.button_clicked :
    fig,ax = plt.subplots(1,1)
    ax.scatter(X[:,0],X[:,1],c=y)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

    st.pyplot(fig=fig)
else:
    params = st.session_state.params
    dt= DecisionTreeClassifier(max_depth=params["max_depth"],criterion=params["criterion"],splitter=params["splitter"],max_features=params["max_feature"],random_state=32,min_samples_leaf=params["min_samples_leaf"],min_samples_split=params["min_sample_split"])
    dt.fit(X_train,y_train)
    y_pred = dt.predict(X_test)
     
    fig1,ax1 = plt.subplots(1,1)
    display=DecisionBoundaryDisplay.from_estimator(dt,X,response_method="predict",ax=ax1)
    
    st.write("accruacy : ",accuracy_score(y_test,y_pred))
    ax1.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
    ax1.set_xlabel("Feature 0")
    ax1.set_ylabel("Feature 1")
 
    st.pyplot(fig=fig1)
    
    fig2,ax2 = plt.subplots(1,1,figsize=[15,10])
    plot_tree(
        dt,ax=ax2,filled=True
    )
    st.pyplot(fig=fig2)
    