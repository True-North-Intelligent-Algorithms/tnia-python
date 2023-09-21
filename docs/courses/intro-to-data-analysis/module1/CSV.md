---
layout: basic
---

### Open a CSV File in a Test Editor

https://www.howtogeek.com/348960/what-is-a-csv-file-and-how-do-i-open-it/

### Open CSV in Python

[See this example](https://github.com/bnorthan/inf-428-data-analytics-online/blob/master/python/notebooks/introduction/OpenCSV.ipynb)  

We can use the Pandas libarary (which we will learn more about in Module 3) to open up a csv file with one line of code

``` python
import pandas as pd  
data=pd.read_csv('donuts.csv')  

# the head function display the first few lines of the data from the csv file  
data.head()  

```

