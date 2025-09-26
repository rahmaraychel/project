# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

stud_mark=pd.read_csv('student_exam_scores.csv')
print(stud_mark.head())
print(stud_mark.isnull().sum)

X=stud_mark[["hours_studied"]]
y=stud_mark["exam_score"]
X_train,x_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=40
)
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(x_test)
print("\n model trained succesfully")
print("R2 scores",r2_score(y_test,y_pred))
# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

stud_mark=pd.read_csv('student_exam_scores.csv')
print(stud_mark.head())
print(stud_mark.isnull().sum)

X=stud_mark[["hours_studied"]]
y=stud_mark["exam_score"]
X_train,x_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=40
)
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(x_test)
print("\n model trained succesfully")
print("R2 scores",r2_score(y_test,y_pred))
# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

stud_mark=pd.read_csv('student_exam_scores.csv')
print(stud_mark.head())
print(stud_mark.isnull().sum)

X=stud_mark[["hours_studied"]]
y=stud_mark["exam_score"]
X_train,x_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=40
)
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(x_test)
print("\n model trained succesfully")
print("R2 scores",r2_score(y_test,y_pred))
# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

stud_mark=pd.read_csv('student_exam_scores.csv')
print(stud_mark.head())
print(stud_mark.isnull().sum)

X=stud_mark[["hours_studied"]]
y=stud_mark["exam_score"]
X_train,x_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=40
)
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(x_test)
print("\n model trained succesfully")
print("R2 scores",r2_score(y_test,y_pred))
predicted_score = model.predict([[3]])
new_data = pd.DataFrame({'hours_studied': [3]})
print(f"Predicted score : {predicted_score[0]:.2f}")

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Study Hours vs Exam Score')
plt.legend()







































































